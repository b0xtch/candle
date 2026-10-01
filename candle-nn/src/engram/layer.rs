//! The Engram module (section 2.3 of the paper): the retrieved memory `e_t` is gated by the
//! current hidden state and refined by a short depthwise causal convolution,
//!
//! ```text
//! k_t = W_K e_t,   v_t = W_V e_t,   α_t = σ(signed_sqrt(RMSNorm(h_t) · RMSNorm(k_t) / √d))
//! Y = SiLU(Conv1D(RMSNorm(α ⊙ V))) + α ⊙ V
//! ```
//!
//! and its output is added to the residual stream before the attention of its decoder block.
use super::config::EngramConfig;
use super::hashing::LayerHashParams;
use super::table::{MemoryTable, PendingRows, Placement, TableOptions};
use crate::{Init, Linear, VarBuilder};
use candle::{DType, Device, Result, Tensor, D};

/// Options used when creating or loading Engram modules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EngramOptions {
    /// Storage of the embedding tables.
    pub table: TableOptions,
    /// Initialize new (i.e. not loaded) value projections to zero. The modules then start as an
    /// exact identity, so that Engram can be attached to a pretrained backbone and trained
    /// without perturbing it at first.
    pub zero_init_value: bool,
}

/// RMS norms with one weight vector per residual branch, applied to `(..., M, D)` tensors.
#[derive(Debug, Clone)]
struct BranchNorm {
    weights: Vec<Tensor>,
    eps: f64,
    /// The `(M, D)` stacked weights, when they are not trained.
    stacked: Option<Tensor>,
    /// `(D,)` ones, for the fused normalization of multi-branch inputs.
    ones: Option<Tensor>,
}

impl BranchNorm {
    fn new(hc_mult: usize, dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weights = (0..hc_mult)
            .map(|m| vb.pp(m).get_with_hints(dim, "weight", Init::Const(1.)))
            .collect::<Result<Vec<_>>>()?;
        let (stacked, ones) = if weights.iter().any(|w| w.track_op()) {
            (None, None)
        } else {
            let ones = if hc_mult > 1 {
                Some(Tensor::ones(dim, vb.dtype(), vb.device())?)
            } else {
                None
            };
            (Some(Tensor::stack(&weights, 0)?), ones)
        };
        Ok(Self {
            weights,
            eps,
            stacked,
            ones,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match (&self.stacked, &self.ones) {
            (Some(_), None) => {
                crate::ops::rms_norm(&xs.contiguous()?, &self.weights[0], self.eps as f32)
            }
            (Some(weights), Some(ones)) => {
                crate::ops::rms_norm(&xs.contiguous()?, ones, self.eps as f32)?
                    .broadcast_mul(weights)
            }
            // The fused kernel has no backward pass.
            (None, _) => {
                let dtype = xs.dtype();
                let internal_dtype = match dtype {
                    DType::F16 | DType::BF16 => DType::F32,
                    dtype => dtype,
                };
                let xs = xs.to_dtype(internal_dtype)?;
                let ms = xs.sqr()?.mean_keepdim(D::Minus1)?;
                let xs = xs.broadcast_div(&(ms + self.eps)?.sqrt()?)?;
                xs.to_dtype(dtype)?
                    .broadcast_mul(&Tensor::stack(&self.weights, 0)?)
            }
        }
    }
}

/// Depthwise causal convolution over the concatenated branches, preceded by per-branch RMS
/// norms and optionally followed by SiLU (`ShortConv` in the reference implementation).
#[derive(Debug, Clone)]
struct ShortConv {
    /// `(channels, 1, kernel_size)`, the layout of a depthwise `torch.nn.Conv1d`.
    weight: Tensor,
    /// The `(channels,)` weights of each tap, when they are not trained.
    taps: Option<Vec<Tensor>>,
    norm: BranchNorm,
    kernel_size: usize,
    dilation: usize,
    activation: bool,
}

impl ShortConv {
    fn new(cfg: &EngramConfig, hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let channels = cfg.hc_mult * hidden_size;
        let kernel_size = cfg.kernel_size;
        // Zero initialization makes the convolution branch vanish at the start of training.
        let weight =
            vb.pp("conv")
                .get_with_hints((channels, 1, kernel_size), "weight", Init::Const(0.))?;
        let taps = if weight.track_op() {
            None
        } else {
            Some(Self::taps(&weight, kernel_size)?)
        };
        let norm = BranchNorm::new(cfg.hc_mult, hidden_size, cfg.conv_norm_eps, vb.pp("norms"))?;
        Ok(Self {
            weight,
            taps,
            norm,
            kernel_size,
            dilation: cfg.max_ngram_size,
            activation: cfg.conv_activation,
        })
    }

    fn taps(weight: &Tensor, kernel_size: usize) -> Result<Vec<Tensor>> {
        (0..kernel_size)
            .map(|j| weight.narrow(2, j, 1)?.flatten_all())
            .collect()
    }

    /// Number of past positions the convolution reads.
    fn state_len(&self) -> usize {
        (self.kernel_size - 1) * self.dilation
    }

    /// `xs` is `(B, T, M, D)`. `state` holds the normalized inputs of the `state_len()`
    /// positions preceding `xs`, `(B, state_len, M * D)`, with `None` meaning the start of the
    /// sequences. Returns the output and the state for the next call.
    fn forward(&self, xs: &Tensor, state: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let (b, t, m, d) = xs.dims4()?;
        let xs = self.norm.forward(xs)?.reshape((b, t, m * d))?;
        let state_len = self.state_len();
        let xs = match state {
            Some(state) => Tensor::cat(&[state, &xs], 1)?,
            None => xs.pad_with_zeros(1, state_len, 0)?,
        };
        let computed;
        let taps = match &self.taps {
            Some(taps) => taps,
            None => {
                computed = Self::taps(&self.weight, self.kernel_size)?;
                &computed
            }
        };
        // Tap j reads position t - (kernel_size - 1 - j) * dilation.
        let mut ys = xs.narrow(1, 0, t)?.broadcast_mul(&taps[0])?;
        for (j, tap) in taps.iter().enumerate().skip(1) {
            ys = (ys + xs.narrow(1, j * self.dilation, t)?.broadcast_mul(tap)?)?;
        }
        if self.activation {
            ys = ys.silu()?;
        }
        let state = xs.narrow(1, t, state_len)?.force_contiguous()?.detach();
        Ok((ys.reshape((b, t, m, d))?, state))
    }
}

#[derive(Debug, Clone)]
enum Projections {
    /// The value projection and one key projection per branch, as stored in checkpoints.
    Separate { value: Linear, keys: Vec<Linear> },
    /// The same weights concatenated into a single `((1 + M) * D, memory_dim)` projection.
    Fused(Linear),
}

fn linear(in_dim: usize, out_dim: usize, zero_init: bool, vb: VarBuilder) -> Result<Linear> {
    if !zero_init {
        return crate::linear(in_dim, out_dim, vb);
    }
    let weight = vb.get_with_hints((out_dim, in_dim), "weight", Init::Const(0.))?;
    let bias = vb.get_with_hints(out_dim, "bias", Init::Const(0.))?;
    Ok(Linear::new(weight, Some(bias)))
}

/// An Engram conditional-memory module, attached to one decoder block.
///
/// The module looks up the hashed n-gram embeddings of every position in its [`MemoryTable`],
/// gates them with the hidden state and returns an update for the residual stream. Most users
/// will drive it through [`super::EngramStack`], which also handles hashing and the state
/// carried across calls when decoding.
#[derive(Debug, Clone)]
pub struct Engram {
    layer_id: usize,
    hidden_size: usize,
    hc_mult: usize,
    memory_dim: usize,
    table: MemoryTable,
    projections: Projections,
    key_norm: BranchNorm,
    query_norm: BranchNorm,
    conv: ShortConv,
}

impl Engram {
    /// Loads the module of decoder block `params.layer_id` or, with a `VarMap`-backed builder,
    /// creates it for training.
    ///
    /// Weights are named as in the reference implementation:
    /// `multi_head_embedding.embedding.weight` (the concatenated tables of all heads),
    /// `value_proj`, `key_projs.{m}`, `norm1.{m}` (keys), `norm2.{m}` (queries),
    /// `short_conv.conv.weight` and `short_conv.norms.{m}`. The embedding table is stored as
    /// requested by `options.table`; when it is kept off the compute device or compressed, it is
    /// loaded on the CPU and never materialized on the device.
    pub fn new(
        cfg: &EngramConfig,
        params: &LayerHashParams,
        hidden_size: usize,
        options: &EngramOptions,
        vb: VarBuilder,
    ) -> Result<Self> {
        let table = Self::load_table(
            cfg,
            params,
            &options.table,
            vb.pp("multi_head_embedding").pp("embedding"),
        )?;
        Self::with_table(cfg, params, hidden_size, table, options, vb)
    }

    fn load_table(
        cfg: &EngramConfig,
        params: &LayerHashParams,
        options: &TableOptions,
        vb: VarBuilder,
    ) -> Result<MemoryTable> {
        let shape = (params.num_rows as usize, cfg.head_dim());
        let init = Init::Randn {
            mean: 0.,
            stdev: 1.,
        };
        if options.placement == Placement::Device && options.compression.is_none() {
            return MemoryTable::on_device(vb.get_with_hints(shape, "weight", init)?);
        }
        let (device, dtype) = (vb.device().clone(), vb.dtype());
        let table = vb
            .set_device(Device::Cpu)
            .get_with_hints(shape, "weight", init)?;
        MemoryTable::from_tensor(table, options, &device, dtype)
    }

    /// Like [`Engram::new`], with an embedding table built separately, e.g. memory mapped from
    /// a file with [`super::MmapRows`]. The table must return rows in the dtype of `vb`.
    pub fn with_table(
        cfg: &EngramConfig,
        params: &LayerHashParams,
        hidden_size: usize,
        table: MemoryTable,
        options: &EngramOptions,
        vb: VarBuilder,
    ) -> Result<Self> {
        cfg.validate()?;
        if table.num_rows() as u64 != params.num_rows || table.row_dim() != cfg.head_dim() {
            candle::bail!(
                "engram: layer {} expects a ({}, {}) table, got ({}, {})",
                params.layer_id,
                params.num_rows,
                cfg.head_dim(),
                table.num_rows(),
                table.row_dim()
            )
        }
        let memory_dim = cfg.memory_dim();
        let hc_mult = cfg.hc_mult;
        let value = linear(
            memory_dim,
            hidden_size,
            options.zero_init_value,
            vb.pp("value_proj"),
        )?;
        let keys = (0..hc_mult)
            .map(|m| crate::linear(memory_dim, hidden_size, vb.pp("key_projs").pp(m)))
            .collect::<Result<Vec<_>>>()?;
        let trained = std::iter::once(&value)
            .chain(keys.iter())
            .any(|l| l.weight().track_op() || l.bias().is_some_and(|b| b.track_op()));
        let projections = if trained {
            Projections::Separate { value, keys }
        } else {
            let layers: Vec<&Linear> = std::iter::once(&value).chain(keys.iter()).collect();
            let weights: Vec<&Tensor> = layers.iter().map(|l| l.weight()).collect();
            let biases = layers
                .iter()
                .map(|l| l.bias().cloned())
                .collect::<Option<Vec<_>>>();
            let bias = match biases {
                Some(biases) => Some(Tensor::cat(&biases, 0)?),
                None => None,
            };
            Projections::Fused(Linear::new(Tensor::cat(&weights, 0)?, bias))
        };
        let eps = cfg.qk_norm_eps();
        Ok(Self {
            layer_id: params.layer_id,
            hidden_size,
            hc_mult,
            memory_dim,
            table,
            projections,
            key_norm: BranchNorm::new(hc_mult, hidden_size, eps, vb.pp("norm1"))?,
            query_norm: BranchNorm::new(hc_mult, hidden_size, eps, vb.pp("norm2"))?,
            conv: ShortConv::new(cfg, hidden_size, vb.pp("short_conv"))?,
        })
    }

    /// Index of the decoder block this module is attached to.
    pub fn layer_id(&self) -> usize {
        self.layer_id
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Number of residual branches.
    pub fn hc_mult(&self) -> usize {
        self.hc_mult
    }

    /// Dimension of the retrieved memory `e_t`.
    pub fn memory_dim(&self) -> usize {
        self.memory_dim
    }

    pub fn table(&self) -> &MemoryTable {
        &self.table
    }

    /// Number of past positions kept in the convolution state.
    pub fn conv_state_len(&self) -> usize {
        self.conv.state_len()
    }

    /// Gathers the memory `e_t`, `(batch, seq_len, memory_dim)`, given the table rows of every
    /// position as computed by [`super::NgramHasher::row_ids`].
    pub fn retrieve(&self, row_ids: &[u32], batch: usize, seq_len: usize) -> Result<Tensor> {
        self.table
            .lookup(row_ids)?
            .reshape((batch, seq_len, self.memory_dim))
    }

    /// Starts gathering the given rows ahead of time, see [`MemoryTable::prefetch`]. The rows
    /// returned by [`PendingRows::wait`] reshape to `(batch, seq_len, memory_dim)`.
    pub fn prefetch(&self, row_ids: Vec<u32>) -> Result<PendingRows> {
        self.table.prefetch(row_ids)
    }

    /// Computes the module output `Y`, to be added to the residual stream, and the new
    /// convolution state.
    ///
    /// * `hidden`: the hidden states `(B, T, D)`, or `(B, T, M, D)` for a backbone with `M`
    ///   residual branches; `Y` has the same shape.
    /// * `memory`: the retrieved memory `(B, T, memory_dim)`, see [`Engram::retrieve`].
    /// * `conv_state`: the state returned by the previous call on the same sequences, or `None`
    ///   at the start of the sequences.
    pub fn forward(
        &self,
        hidden: &Tensor,
        memory: &Tensor,
        conv_state: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let single_branch = hidden.rank() == 3;
        let hidden = if single_branch {
            hidden.unsqueeze(2)?
        } else {
            hidden.clone()
        };
        let (b, t, m, d) = hidden.dims4()?;
        if m != self.hc_mult || d != self.hidden_size {
            candle::bail!(
                "engram: expected hidden states with {} branches of size {}, got {:?}",
                self.hc_mult,
                self.hidden_size,
                hidden.shape()
            )
        }
        let (value, keys) = match &self.projections {
            Projections::Fused(proj) => {
                let kv = memory.apply(proj)?;
                let keys = kv.narrow(D::Minus1, d, m * d)?.reshape((b, t, m, d))?;
                (kv.narrow(D::Minus1, 0, d)?, keys)
            }
            Projections::Separate { value, keys } => {
                let keys = keys
                    .iter()
                    .map(|k| memory.apply(k))
                    .collect::<Result<Vec<_>>>()?;
                (memory.apply(value)?, Tensor::stack(&keys, 2)?)
            }
        };
        let keys = self.key_norm.forward(&keys)?.to_dtype(DType::F32)?;
        let queries = self.query_norm.forward(&hidden)?.to_dtype(DType::F32)?;
        // α = σ(signed_sqrt(<k, q> / √d)), in f32 whatever the model dtype.
        let gate = ((keys * queries)?.sum_keepdim(D::Minus1)? / (d as f64).sqrt())?;
        let gate = (gate.abs()?.maximum(1e-6)?.sqrt()? * gate.sign()?)?;
        let gate = crate::ops::sigmoid(&gate)?.to_dtype(value.dtype())?;
        let gated = gate.broadcast_mul(&value.unsqueeze(2)?)?;
        let (conv, conv_state) = self.conv.forward(&gated, conv_state)?;
        let ys = (gated + conv)?;
        let ys = if single_branch { ys.squeeze(2)? } else { ys };
        Ok((ys, conv_state))
    }
}
