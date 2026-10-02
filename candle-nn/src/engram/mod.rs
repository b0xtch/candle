//! Engram: conditional memory via scalable n-gram lookup, for any decoder.
//!
//! Engram ("Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language
//! Models", Cheng et al., 2026, <https://github.com/deepseek-ai/Engram>) adds a large table of
//! hashed n-gram embeddings to a transformer. At a few decoder blocks, the embeddings of the
//! suffix n-grams ending at each position are looked up in O(1), gated by the current hidden
//! state, optionally refined by a short causal convolution and added to the residual stream.
//! Because the addresses only depend on the token ids, the tables can be kept off the
//! accelerator and their rows fetched ahead of time.
//!
//! This module implements the whole pipeline in a model-agnostic way:
//!
//! - [`VocabProjection`]: tokenizer compression, i.e. the surjective map from raw token ids to
//!   canonical ids (case, accents and whitespace folded) that n-grams are built from.
//! - [`NgramHasher`]: deterministic multi-head hashing of the 2..N-grams of canonical ids,
//!   bit-compatible with the reference implementation (including its NumPy-seeded
//!   multipliers and prime table sizes), with [`NgramHistory`] for chunked and incremental
//!   decoding.
//! - [`MemoryTable`]: the embedding tables, stored on the compute device (dense, fp16/bf16 or
//!   block-quantized such as `Q8_0`/`Q4_0`), offloaded to host memory (dense, quantized or
//!   MXFP8), or memory mapped from safetensors/GGUF files with [`MmapRows`] so that tables
//!   larger than RAM work and the operating system keeps the Zipfian hot set cached. Offloaded
//!   rows are gathered on a background thread with [`MemoryTable::prefetch`], and only the
//!   gathered rows are moved to the device.
//! - [`Engram`]: the module itself (context-aware gating and optional short convolution), for
//!   backbones with a single residual stream or with `M` hyper-connection branches.
//! - [`EngramStack`]: everything a decoder needs, driven by two calls per forward pass,
//!   [`EngramStack::begin`] and [`EngramStack::apply`].
//!
//! Weights are named as in the reference implementation so that its checkpoints load as-is,
//! and the modules can be trained with a `VarMap` (keep the tables on the device). With
//! [`EngramOptions::zero_init_value`], fresh modules start as an exact identity, which allows
//! attaching Engram to an existing pretrained model and training only the new parameters.
//!
//! The hashing also reproduces the table sizes of DeepSeek-V4.1-Flash, which ships Engram
//! without the convolution (`kernel_size: 0`) and with MXFP8 tables that
//! [`MmapRows::from_safetensors_mxfp8`] maps as they are.
//!
//! ```no_run
//! use candle::{DType, Device, Module, Tensor};
//! use candle_nn::engram::{
//!     EngramConfig, EngramOptions, EngramStack, RowFormat, TableOptions, VocabProjection,
//! };
//! use candle::quantized::GgmlDType;
//!
//! # fn main() -> candle::Result<()> {
//! let device = Device::Cpu;
//! let config = EngramConfig {
//!     engram_vocab_size: vec![100_000, 100_000],
//!     n_embed_per_ngram: 256,
//!     n_head_per_ngram: 4,
//!     layer_ids: vec![1, 6],
//!     ..EngramConfig::reference_demo(1)
//! };
//! // Map raw token ids to canonical ids, here without compression.
//! let projection = VocabProjection::identity(32_000);
//! // Tables offloaded to host memory, stored as Q8_0 and prefetched.
//! let options = EngramOptions {
//!     table: TableOptions::offloaded(Some(RowFormat::Ggml(GgmlDType::Q8_0))),
//!     ..Default::default()
//! };
//! let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
//! let mut engram = EngramStack::load(config, projection, 512, &options, vb.pp("layers"))?;
//!
//! let input_ids = Tensor::new(&[[1u32, 5, 9, 2]], &device)?;
//! engram.begin(&input_ids, 0)?;
//! let mut xs = Tensor::zeros((1, 4, 512), DType::F32, &device)?;
//! for block_idx in 0..8 {
//!     xs = engram.apply(block_idx, &xs)?;
//!     // xs = block.forward(&xs)?;
//! }
//! # Ok(())
//! # }
//! ```
pub mod config;
pub mod hashing;
pub mod layer;
mod numpy_rng;
pub mod stack;
pub mod table;
pub mod vocab;

pub use config::EngramConfig;
pub use hashing::{is_prime, LayerHashParams, NgramHasher, NgramHistory};
pub use layer::{Engram, EngramOptions};
pub use stack::EngramStack;
pub use table::{
    HostQuantizedRows, HostRowStore, HostTensorRows, MemoryTable, MmapRows, Mxfp8Rows, PendingRows,
    Placement, RowFormat, TableOptions,
};
pub use vocab::{reference_normalizer, VocabProjection};
