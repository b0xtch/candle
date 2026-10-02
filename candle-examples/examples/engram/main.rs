#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use candle::quantized::{gguf_file, GgmlDType, QTensor};
use candle::{DType, Device, Shape, Tensor};
use candle_nn::engram::{
    Engram, EngramConfig, EngramOptions, EngramStack, HostRowStore, MemoryTable, MmapRows,
    Mxfp8Rows, NgramHasher, Placement, RowFormat, TableOptions, VocabProjection,
};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::{Init, VarBuilder};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_qwen3::ModelWeights as Qwen3;
use clap::{Parser, ValueEnum};
use std::io::Write;
use std::sync::Arc;
use tokenizers::Tokenizer;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Storage {
    /// On the compute device.
    Device,
    /// In host memory, only the looked up rows are copied to the device.
    Host,
    /// Memory mapped from a file, paged in on demand by the operating system.
    Mmap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Compression {
    None,
    F16,
    Bf16,
    #[value(name = "q8_0")]
    Q8_0,
    #[value(name = "q4_0")]
    Q4_0,
    /// E4M3 values with a power-of-two scale per 32 values, as in DeepSeek-V4.1.
    Mxfp8,
}

impl Compression {
    fn format(&self) -> Option<RowFormat> {
        match self {
            Self::None => None,
            Self::F16 => Some(DType::F16.into()),
            Self::Bf16 => Some(DType::BF16.into()),
            Self::Q8_0 => Some(GgmlDType::Q8_0.into()),
            Self::Q4_0 => Some(GgmlDType::Q4_0.into()),
            Self::Mxfp8 => Some(RowFormat::Mxfp8),
        }
    }

    /// The GGML type of the tables written to GGUF files, `None` for MXFP8 which is written to
    /// safetensors files.
    fn ggml_dtype(&self) -> Option<GgmlDType> {
        match self {
            Self::None => Some(GgmlDType::F32),
            Self::F16 => Some(GgmlDType::F16),
            Self::Bf16 => Some(GgmlDType::BF16),
            Self::Q8_0 => Some(GgmlDType::Q8_0),
            Self::Q4_0 => Some(GgmlDType::Q4_0),
            Self::Mxfp8 => None,
        }
    }

    fn tables_file(&self) -> String {
        match self.ggml_dtype() {
            Some(dtype) => format!("engram-tables-{}.gguf", format!("{dtype:?}").to_lowercase()),
            None => "engram-tables-mxfp8.safetensors".to_string(),
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// GGUF file of a Qwen3 model, defaults to Qwen3-0.6B (Q8_0).
    #[arg(long)]
    model: Option<String>,

    /// The tokenizer config in json format, defaults to the one of Qwen3-0.6B.
    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(long, default_value = "The capital of France is")]
    prompt: String,

    /// The length of the sample to generate (in tokens).
    #[arg(short = 'n', long, default_value_t = 64)]
    sample_len: usize,

    /// The temperature used to generate samples, use 0 for greedy sampling.
    #[arg(long, default_value_t = 0.)]
    temperature: f64,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Engram configuration as json, using the field names of the reference implementation.
    /// Defaults to its demo configuration (2/3-grams, 8 heads, 512 dims per n-gram order) at
    /// blocks 1 and 15, with as many buckets per n-gram order as canonical tokens.
    #[arg(long)]
    engram_config: Option<String>,

    /// Trained Engram weights (safetensors), named `layers.{i}.engram.*`. Without them, fresh
    /// modules are created with zero value projections, which leave the model outputs unchanged.
    #[arg(long)]
    engram_weights: Option<String>,

    /// Where the embedding tables are kept.
    #[arg(long, value_enum, default_value = "host")]
    storage: Storage,

    /// How the embedding tables are encoded.
    #[arg(long, value_enum, default_value = "q8_0")]
    compression: Compression,

    /// File holding the tables for `--storage mmap`, written when it does not exist: GGUF, or
    /// safetensors for MXFP8. Defaults to `engram-tables-{compression}.{gguf,safetensors}`.
    #[arg(long)]
    tables_file: Option<String>,

    /// Gather offloaded rows when the layer needs them instead of ahead of time.
    #[arg(long)]
    no_prefetch: bool,

    /// Also generate without Engram, to compare the speed and the outputs.
    #[arg(long)]
    compare: bool,
}

impl Args {
    fn tokenizer(&self) -> Result<Tokenizer> {
        let path = match &self.tokenizer {
            Some(path) => std::path::PathBuf::from(path),
            None => candle_examples::hub::Api::new()?
                .model("Qwen/Qwen3-0.6B")
                .get("tokenizer.json")?,
        };
        Tokenizer::from_file(path).map_err(E::msg)
    }

    fn model(&self) -> Result<std::path::PathBuf> {
        Ok(match &self.model {
            Some(path) => std::path::PathBuf::from(path),
            None => candle_examples::hub::Api::new()?
                .model("unsloth/Qwen3-0.6B-GGUF")
                .get("Qwen3-0.6B-Q8_0.gguf")?,
        })
    }
}

/// Creates the weights that a checkpoint would otherwise provide, from their initialization
/// hints, as plain (not trainable) tensors.
struct FreshInit;

impl SimpleBackend for FreshInit {
    fn get(
        &self,
        s: Shape,
        _: &str,
        h: Init,
        dtype: DType,
        dev: &Device,
    ) -> candle::Result<Tensor> {
        Ok(h.var(s, dtype, dev)?.as_detached_tensor())
    }

    fn get_unchecked(&self, name: &str, _: DType, _: &Device) -> candle::Result<Tensor> {
        candle::bail!("no shape for {name}")
    }

    fn contains_tensor(&self, _: &str) -> bool {
        true
    }
}

fn format_size(bytes: usize) -> String {
    match bytes {
        b if b < 1 << 20 => format!("{:.1}KiB", b as f64 / 1024.),
        b if b < 1 << 30 => format!("{:.1}MiB", b as f64 / (1 << 20) as f64),
        b => format!("{:.2}GiB", b as f64 / (1 << 30) as f64),
    }
}

fn table_name(layer_id: usize) -> String {
    format!("blk.{layer_id}.engram.embd.weight")
}

fn scale_name(layer_id: usize) -> String {
    format!("blk.{layer_id}.engram.embd.scale")
}

/// Writes the embedding tables of every Engram layer, encoded as `compression`: to a GGUF file,
/// or for MXFP8 to a safetensors file holding the values and the scales of each table.
fn write_tables(
    path: &str,
    config: &EngramConfig,
    hasher: &NgramHasher,
    vb: &VarBuilder,
    compression: Compression,
) -> Result<()> {
    let mut tables = vec![];
    for params in hasher.layers() {
        let table = vb
            .pp(params.layer_id)
            .pp("engram.multi_head_embedding.embedding")
            .set_device(Device::Cpu)
            .get_with_hints(
                (params.num_rows as usize, config.head_dim()),
                "weight",
                Init::Randn {
                    mean: 0.,
                    stdev: 1.,
                },
            )?;
        tables.push((params.layer_id, table));
    }
    match compression.ggml_dtype() {
        Some(dtype) => {
            let tables = tables
                .iter()
                .map(|(id, t)| Ok((table_name(*id), QTensor::quantize(t, dtype)?)))
                .collect::<Result<Vec<_>>>()?;
            let tensors: Vec<(&str, &QTensor)> =
                tables.iter().map(|(n, t)| (n.as_str(), t)).collect();
            let mut file = std::fs::File::create(path)?;
            gguf_file::write(&mut file, &[], &tensors)?;
        }
        None => {
            use safetensors::{tensor::TensorView, Dtype};
            let tables = tables
                .iter()
                .map(|(id, t)| Ok((*id, Mxfp8Rows::quantize(t)?)))
                .collect::<Result<Vec<_>>>()?;
            let mut views = vec![];
            for (id, rows) in tables.iter() {
                let (num_rows, dim) = (rows.num_rows(), rows.row_dim());
                let values = TensorView::new(Dtype::F8_E4M3, vec![num_rows, dim], rows.values())?;
                let scales =
                    TensorView::new(Dtype::F8_E8M0, vec![num_rows, dim / 32], rows.scales())?;
                views.push((table_name(*id), values));
                views.push((scale_name(*id), scales));
            }
            safetensors::serialize_to_file(views, None, std::path::Path::new(path))?;
        }
    }
    Ok(())
}

fn generate(
    model: &mut Qwen3,
    tokens: &[u32],
    args: &Args,
    device: &Device,
    tokenizer: &Tokenizer,
) -> Result<(Vec<u32>, f64, f64)> {
    model.clear_kv_cache();
    let sampling = if args.temperature <= 0. {
        Sampling::ArgMax
    } else {
        Sampling::All {
            temperature: args.temperature,
        }
    };
    let mut logits_processor = LogitsProcessor::from_sampling(args.seed, sampling);
    let eos = tokenizer.token_to_id("<|im_end|>");
    device.with_context(|| -> Result<_> {
        let start = std::time::Instant::now();
        let input = Tensor::new(tokens, device)?.unsqueeze(0)?;
        let logits = model.forward(&input, 0)?.squeeze(0)?;
        let mut next = logits_processor.sample(&logits)?;
        let prompt_dt = start.elapsed().as_secs_f64();
        let mut generated = vec![next];
        let start = std::time::Instant::now();
        while generated.len() < args.sample_len && Some(next) != eos {
            let input = Tensor::new(&[next], device)?.unsqueeze(0)?;
            let pos = tokens.len() + generated.len() - 1;
            let logits = model.forward(&input, pos)?.squeeze(0)?;
            next = logits_processor.sample(&logits)?;
            generated.push(next);
        }
        let tokens_per_s = (generated.len() - 1) as f64 / start.elapsed().as_secs_f64();
        Ok((generated, tokens.len() as f64 / prompt_dt, tokens_per_s))
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
    let tokenizer = args.tokenizer()?;

    // Tokenizer compression: n-grams are hashed over canonical ids.
    let start = std::time::Instant::now();
    let projection = VocabProjection::from_tokenizer(&tokenizer)?;
    println!(
        "tokenizer compression: {} -> {} ids ({:.1}% fewer) in {:.2}s",
        projection.vocab_size(),
        projection.num_compressed(),
        100. * projection.compression_ratio(),
        start.elapsed().as_secs_f32()
    );

    let model_path = args.model()?;
    let mut file = std::fs::File::open(&model_path)?;
    let content = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(&model_path))?;
    let hidden_size = content.metadata["qwen3.embedding_length"].to_u32()? as usize;
    let num_layers = content.metadata["qwen3.block_count"].to_u32()? as usize;
    let mut model = Qwen3::from_gguf(content, &mut file, &device)?;

    let config = match &args.engram_config {
        Some(path) => serde_json::from_str(&std::fs::read_to_string(path)?)?,
        None => EngramConfig {
            engram_vocab_size: vec![projection.num_compressed(); 2],
            layer_ids: vec![1, 15],
            pad_id: tokenizer.token_to_id("<|endoftext|>").unwrap_or(0),
            ..EngramConfig::reference_demo(1)
        },
    };
    if let Some(&id) = config.layer_ids.iter().find(|&&id| id >= num_layers) {
        anyhow::bail!("engram layer {id} is past the last of the {num_layers} blocks")
    }

    // The quantized Qwen3 model keeps its hidden states in f32.
    let vb = match &args.engram_weights {
        Some(path) => unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &device)? },
        None => VarBuilder::from_backend(Box::new(FreshInit), DType::F32, device.clone()),
    };
    let vb = vb.pp("layers");
    let options = EngramOptions {
        table: TableOptions {
            placement: match args.storage {
                Storage::Device => Placement::Device,
                Storage::Host | Storage::Mmap => Placement::Host,
            },
            compression: args.compression.format(),
            prefetch: !args.no_prefetch,
        },
        zero_init_value: args.engram_weights.is_none(),
    };
    let start = std::time::Instant::now();
    let engram = if args.storage == Storage::Mmap {
        let hasher = NgramHasher::with_projection(&config, &projection)?;
        let tables_file = match &args.tables_file {
            Some(path) => path.clone(),
            None => args.compression.tables_file(),
        };
        if !std::path::Path::new(&tables_file).exists() {
            println!("writing the tables to {tables_file}");
            write_tables(&tables_file, &config, &hasher, &vb, args.compression)?;
        }
        let modules = hasher
            .layers()
            .iter()
            .map(|params| {
                let id = params.layer_id;
                let rows = unsafe {
                    match args.compression {
                        Compression::Mxfp8 => MmapRows::from_safetensors_mxfp8(
                            &tables_file,
                            &table_name(id),
                            &scale_name(id),
                        )?,
                        _ => MmapRows::from_gguf(&tables_file, &table_name(id))?,
                    }
                };
                let table = MemoryTable::offloaded(Arc::new(rows), &device, DType::F32)
                    .with_prefetch(!args.no_prefetch);
                let vb = vb.pp(params.layer_id).pp("engram");
                Engram::with_table(&config, params, hidden_size, table, &options, vb)
            })
            .collect::<candle::Result<Vec<_>>>()?;
        EngramStack::new(config, projection, modules)?
    } else {
        EngramStack::load(config, projection, hidden_size, &options, vb)?
    };
    let (device_bytes, total_bytes) = engram.table_bytes();
    println!(
        "engram built in {:.2}s: {} of tables, {} on the compute device",
        start.elapsed().as_secs_f32(),
        format_size(total_bytes),
        format_size(device_bytes),
    );
    for module in engram.modules() {
        let table = module.table();
        println!(
            "  block {:2}: {} rows x {} ({}, {})",
            module.layer_id(),
            table.num_rows(),
            table.row_dim(),
            table.describe(),
            format_size(table.storage_bytes()),
        );
    }

    let prompt = format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        args.prompt
    );
    let tokens = tokenizer.encode(prompt, true).map_err(E::msg)?;
    let tokens = tokens.get_ids();

    let baseline = if args.compare {
        let (generated, prompt_speed, speed) =
            generate(&mut model, tokens, &args, &device, &tokenizer)?;
        println!("without engram: prompt {prompt_speed:.1} token/s, generation {speed:.1} token/s");
        Some((generated, speed))
    } else {
        None
    };

    model.set_engram(Some(engram));
    let (generated, prompt_speed, speed) =
        generate(&mut model, tokens, &args, &device, &tokenizer)?;
    println!("with engram:    prompt {prompt_speed:.1} token/s, generation {speed:.1} token/s");
    if let Some((base_tokens, base_speed)) = baseline {
        println!(
            "generation overhead: {:.1}%, same tokens: {}",
            100. * (base_speed / speed - 1.),
            base_tokens == generated
        );
    }
    let text = tokenizer.decode(&generated, true).map_err(E::msg)?;
    println!("\n{}{text}", args.prompt);
    std::io::stdout().flush()?;
    Ok(())
}
