use candle::Result;

fn default_max_ngram_size() -> usize {
    3
}
fn default_n_embed_per_ngram() -> usize {
    512
}
fn default_n_head_per_ngram() -> usize {
    8
}
fn default_kernel_size() -> usize {
    4
}
fn default_hc_mult() -> usize {
    1
}
fn default_conv_norm_eps() -> f64 {
    1e-5
}
fn default_true() -> bool {
    true
}

/// Configuration of the Engram conditional-memory modules attached to a backbone.
///
/// Field names and defaults follow `EngramConfig` in the reference implementation
/// (<https://github.com/deepseek-ai/Engram>) so that a JSON dump of the reference config
/// deserializes as-is, with a few additions for backbones other than the paper's
/// hyper-connected one (`hc_mult`, the norm epsilons, `conv_activation`).
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct EngramConfig {
    /// Base number of hash buckets for each n-gram order `2..=max_ngram_size`, one entry per
    /// order. Every hash head gets a distinct prime table size greater than or equal to this.
    pub engram_vocab_size: Vec<usize>,
    /// Largest n-gram order, `N`. Orders `2..=N` are hashed.
    #[serde(default = "default_max_ngram_size")]
    pub max_ngram_size: usize,
    /// Memory dimension contributed by each n-gram order, split evenly across its heads.
    #[serde(default = "default_n_embed_per_ngram")]
    pub n_embed_per_ngram: usize,
    /// Number of hash heads, i.e. independent tables, per n-gram order.
    #[serde(default = "default_n_head_per_ngram")]
    pub n_head_per_ngram: usize,
    /// Indexes of the decoder blocks that get an Engram module. The module runs on the
    /// residual stream before the block's attention. The ids also seed the per-layer hash
    /// functions, so they are part of a checkpoint's identity.
    pub layer_ids: Vec<usize>,
    /// Raw (uncompressed) id of the padding token used for positions before the start of the
    /// sequence.
    #[serde(default)]
    pub pad_id: u32,
    /// Seed of the hash multipliers.
    #[serde(default)]
    pub seed: u64,
    /// Kernel size of the depthwise causal convolution (its dilation is `max_ngram_size`). `0`
    /// removes the convolution, as in DeepSeek-V4.1, so that the module returns the gated value.
    #[serde(default = "default_kernel_size")]
    pub kernel_size: usize,
    /// Number of residual branches (`M` in the paper). Standard transformers use 1, the paper
    /// uses 4 with manifold-constrained hyper-connections.
    #[serde(default = "default_hc_mult")]
    pub hc_mult: usize,
    /// Epsilon of the query/key RMS norms. `None` uses `f32::EPSILON`, which matches the
    /// `torch.nn.RMSNorm` default used by the reference for float32 weights.
    #[serde(default)]
    pub norm_eps: Option<f64>,
    /// Epsilon of the RMS norms that precede the short convolution.
    #[serde(default = "default_conv_norm_eps")]
    pub conv_norm_eps: f64,
    /// Whether the short convolution is followed by a SiLU activation.
    #[serde(default = "default_true")]
    pub conv_activation: bool,
}

impl EngramConfig {
    /// The configuration of the reference demo, for a backbone with the given number of
    /// residual branches.
    pub fn reference_demo(hc_mult: usize) -> Self {
        Self {
            engram_vocab_size: vec![129280 * 5, 129280 * 5],
            max_ngram_size: 3,
            n_embed_per_ngram: 512,
            n_head_per_ngram: 8,
            layer_ids: vec![1, 15],
            pad_id: 2,
            seed: 0,
            kernel_size: 4,
            hc_mult,
            norm_eps: None,
            conv_norm_eps: 1e-5,
            conv_activation: true,
        }
    }

    /// Number of hash heads per position, `(max_ngram_size - 1) * n_head_per_ngram`.
    pub fn num_heads(&self) -> usize {
        (self.max_ngram_size - 1) * self.n_head_per_ngram
    }

    /// Dimension of each head's embedding rows.
    pub fn head_dim(&self) -> usize {
        self.n_embed_per_ngram / self.n_head_per_ngram
    }

    /// Dimension of the concatenated memory vector `e_t`.
    pub fn memory_dim(&self) -> usize {
        (self.max_ngram_size - 1) * self.n_embed_per_ngram
    }

    pub fn qk_norm_eps(&self) -> f64 {
        self.norm_eps.unwrap_or(f32::EPSILON as f64)
    }

    pub fn validate(&self) -> Result<()> {
        if self.max_ngram_size < 2 {
            candle::bail!(
                "engram: max_ngram_size must be at least 2, got {}",
                self.max_ngram_size
            )
        }
        if self.engram_vocab_size.len() != self.max_ngram_size - 1 {
            candle::bail!(
                "engram: engram_vocab_size needs one entry per n-gram order 2..={}, got {}",
                self.max_ngram_size,
                self.engram_vocab_size.len()
            )
        }
        if self.engram_vocab_size.contains(&0) {
            candle::bail!("engram: engram_vocab_size entries must be positive")
        }
        if self.n_head_per_ngram == 0
            || !self.n_embed_per_ngram.is_multiple_of(self.n_head_per_ngram)
        {
            candle::bail!(
                "engram: n_embed_per_ngram ({}) must be a positive multiple of n_head_per_ngram ({})",
                self.n_embed_per_ngram,
                self.n_head_per_ngram
            )
        }
        if self.n_embed_per_ngram == 0 {
            candle::bail!("engram: n_embed_per_ngram must be positive")
        }
        if self.hc_mult == 0 {
            candle::bail!("engram: hc_mult must be positive")
        }
        let mut ids = self.layer_ids.clone();
        ids.sort_unstable();
        ids.dedup();
        if ids.len() != self.layer_ids.len() {
            candle::bail!(
                "engram: duplicate entries in layer_ids {:?}",
                self.layer_ids
            )
        }
        Ok(())
    }
}
