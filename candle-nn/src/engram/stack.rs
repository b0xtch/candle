//! All the Engram modules of a backbone, with the tokenizer compression, the hashing and the
//! per-sequence state they share.
use super::config::EngramConfig;
use super::hashing::{NgramHasher, NgramHistory};
use super::layer::{Engram, EngramOptions};
use super::table::PendingRows;
use super::vocab::VocabProjection;
use crate::VarBuilder;
use candle::{DType, Device, Result, Tensor};

/// Engram for a whole decoder: tokenizer compression, n-gram hashing, the per-layer modules and
/// the state carried across calls (the last tokens of every sequence and the convolution
/// states), much like a KV cache.
///
/// Integrating it in a decoder takes two calls per forward pass:
///
/// ```ignore
/// // Hash the n-grams of the new tokens and start retrieving the memory of every Engram layer;
/// // offloaded tables are read in the background while the first blocks run.
/// engram.begin(&input_ids, seqlen_offset)?;
/// let mut xs = embed_tokens.forward(&input_ids)?;
/// for (block_idx, block) in blocks.iter_mut().enumerate() {
///     // H <- H + Engram(H) for blocks that have a module, identity otherwise.
///     xs = engram.apply(block_idx, &xs)?;
///     xs = block.forward(&xs, ...)?;
/// }
/// ```
#[derive(Debug)]
pub struct EngramStack {
    config: EngramConfig,
    projection: VocabProjection,
    hasher: NgramHasher,
    modules: Vec<Engram>,
    history: Option<NgramHistory>,
    position: usize,
    conv_states: Vec<Option<Tensor>>,
    pending: Vec<Option<PendingRows>>,
    step: Option<(usize, usize)>,
}

impl Clone for EngramStack {
    /// Clones the modules and the sequence state; retrievals started by [`EngramStack::begin`]
    /// are not carried over.
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            projection: self.projection.clone(),
            hasher: self.hasher.clone(),
            modules: self.modules.clone(),
            history: self.history.clone(),
            position: self.position,
            conv_states: self.conv_states.clone(),
            pending: self.modules.iter().map(|_| None).collect(),
            step: None,
        }
    }
}

impl EngramStack {
    /// The hasher for `config`, with the padding id mapped through `projection`.
    pub fn hasher(config: &EngramConfig, projection: &VocabProjection) -> Result<NgramHasher> {
        let pad_id = projection.project(config.pad_id)?;
        NgramHasher::new(config, projection.num_compressed(), pad_id)
    }

    /// Groups modules built with [`Engram::new`], one per entry of `config.layer_ids`.
    pub fn new(
        config: EngramConfig,
        projection: VocabProjection,
        modules: Vec<Engram>,
    ) -> Result<Self> {
        let hasher = Self::hasher(&config, &projection)?;
        let mut layer_ids: Vec<usize> = modules.iter().map(|m| m.layer_id()).collect();
        layer_ids.sort_unstable();
        let mut expected = config.layer_ids.clone();
        expected.sort_unstable();
        if layer_ids != expected {
            candle::bail!(
                "engram: got modules for layers {layer_ids:?}, the config has {:?}",
                config.layer_ids
            )
        }
        let n = modules.len();
        Ok(Self {
            config,
            projection,
            hasher,
            modules,
            history: None,
            position: 0,
            conv_states: vec![None; n],
            pending: (0..n).map(|_| None).collect(),
            step: None,
        })
    }

    /// Loads (or creates, with a `VarMap`) the module of every layer in `config.layer_ids`.
    ///
    /// `vb` points at the decoder blocks: the module of block `i` is read from
    /// `vb.pp(i).pp("engram")`, i.e. `layers.{i}.engram.*` as in the reference implementation.
    pub fn load(
        config: EngramConfig,
        projection: VocabProjection,
        hidden_size: usize,
        options: &EngramOptions,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hasher = Self::hasher(&config, &projection)?;
        let modules = hasher
            .layers()
            .iter()
            .map(|params| {
                let vb = vb.pp(params.layer_id).pp("engram");
                Engram::new(&config, params, hidden_size, options, vb)
            })
            .collect::<Result<Vec<_>>>()?;
        Self::new(config, projection, modules)
    }

    pub fn config(&self) -> &EngramConfig {
        &self.config
    }

    pub fn projection(&self) -> &VocabProjection {
        &self.projection
    }

    pub fn ngram_hasher(&self) -> &NgramHasher {
        &self.hasher
    }

    pub fn modules(&self) -> &[Engram] {
        &self.modules
    }

    /// The module attached to decoder block `block_idx`, if any.
    pub fn module(&self, block_idx: usize) -> Option<&Engram> {
        self.modules.iter().find(|m| m.layer_id() == block_idx)
    }

    /// Number of tokens processed since the last reset.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Bytes of embedding tables held on the compute device and in total.
    pub fn table_bytes(&self) -> (usize, usize) {
        self.modules.iter().fold((0, 0), |(device, total), m| {
            (
                device + m.table().device_bytes(),
                total + m.table().storage_bytes(),
            )
        })
    }

    /// Forgets the processed tokens; to be called whenever the KV cache is cleared.
    pub fn reset(&mut self) {
        self.history = None;
        self.position = 0;
        self.conv_states.iter_mut().for_each(|s| *s = None);
        self.pending.iter_mut().for_each(|p| *p = None);
        self.step = None;
    }

    /// Starts a forward pass on `input_ids`, `(batch, seq_len)` raw token ids that follow the
    /// `seqlen_offset` tokens already processed (`0` starts new sequences).
    ///
    /// The n-grams ending at every new position are hashed for all the Engram layers at once
    /// and their memory starts being retrieved, so that the retrieval from offloaded tables
    /// overlaps with the computation of the blocks preceding each Engram layer.
    pub fn begin(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<()> {
        let (batch, seq_len) = input_ids.dims2()?;
        let ids = input_ids
            .flatten_all()?
            .to_device(&Device::Cpu)?
            .to_dtype(DType::U32)?
            .to_vec1::<u32>()?;
        self.begin_ids(&ids, batch, seq_len, seqlen_offset)
    }

    /// [`EngramStack::begin`] with the token ids as a `(batch, seq_len)` row-major slice.
    pub fn begin_ids(
        &mut self,
        ids: &[u32],
        batch: usize,
        seq_len: usize,
        seqlen_offset: usize,
    ) -> Result<()> {
        if ids.len() != batch * seq_len {
            candle::bail!(
                "engram: got {} token ids for a batch of {batch} sequences of length {seq_len}",
                ids.len()
            )
        }
        if seqlen_offset == 0 {
            self.reset();
            self.history = Some(self.hasher.new_history(batch));
        }
        let history = match &mut self.history {
            Some(h) if h.batch_size() == batch && self.position == seqlen_offset => h,
            _ => candle::bail!(
                "engram: cannot continue at offset {seqlen_offset} with a batch of {batch}, \
                 {} tokens of {} sequences were processed (offsets must follow each other, \
                 use offset 0 to start new sequences)",
                self.position,
                self.history.as_ref().map_or(0, |h| h.batch_size()),
            ),
        };
        let ids = self.projection.project_all(ids)?;
        for (module, pending) in self.modules.iter().zip(self.pending.iter_mut()) {
            let rows = self
                .hasher
                .row_ids(module.layer_id(), &ids, seq_len, history)?;
            *pending = Some(module.prefetch(rows)?);
        }
        history.push(&ids, seq_len);
        self.position += seq_len;
        self.step = Some((batch, seq_len));
        Ok(())
    }

    /// Applies the Engram module of decoder block `block_idx` to the residual stream `xs`,
    /// `(B, T, D)` or `(B, T, M, D)`, returning `xs + Engram(xs)`. Blocks without a module return
    /// `xs` unchanged. Each module is applied once per [`EngramStack::begin`].
    pub fn apply(&mut self, block_idx: usize, xs: &Tensor) -> Result<Tensor> {
        let Some(i) = self.modules.iter().position(|m| m.layer_id() == block_idx) else {
            return Ok(xs.clone());
        };
        let (Some((batch, seq_len)), Some(pending)) = (self.step, self.pending[i].take()) else {
            candle::bail!(
                "engram: layer {block_idx} has no pending memory, call begin before each forward pass"
            )
        };
        if xs.dim(0)? != batch || xs.dim(1)? != seq_len {
            candle::bail!(
                "engram: begin was called for ({batch}, {seq_len}) tokens, got hidden states {:?}",
                xs.shape()
            )
        }
        let module = &self.modules[i];
        let memory = pending
            .wait()?
            .reshape((batch, seq_len, module.memory_dim()))?;
        let (ys, state) = module.forward(xs, &memory, self.conv_states[i].as_ref())?;
        self.conv_states[i] = Some(state);
        xs + ys
    }
}
