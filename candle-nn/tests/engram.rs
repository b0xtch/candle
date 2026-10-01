#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::quantized::{gguf_file, GgmlDType, QTensor};
use candle::{DType, Device, Result, Tensor};
use candle_nn::engram::vocab::normalize;
use candle_nn::engram::{
    reference_normalizer, Engram, EngramConfig, EngramOptions, EngramStack, HostQuantizedRows,
    MemoryTable, MmapRows, Placement, TableOptions, VocabProjection,
};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use std::collections::HashMap;
use std::sync::Arc;

// Produced by `engram_reference.py` with the official PyTorch demo of DeepSeek's Engram.
const FIXTURE: &str = "tests/engram_reference.safetensors";

struct Case {
    name: &'static str,
    config: EngramConfig,
    hidden_size: usize,
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "mhc",
            config: EngramConfig {
                engram_vocab_size: vec![61, 53],
                max_ngram_size: 3,
                n_embed_per_ngram: 8,
                n_head_per_ngram: 2,
                layer_ids: vec![1, 3],
                pad_id: 2,
                seed: 0,
                kernel_size: 4,
                hc_mult: 2,
                norm_eps: None,
                conv_norm_eps: 1e-5,
                conv_activation: true,
            },
            hidden_size: 16,
        },
        Case {
            name: "single",
            config: EngramConfig {
                engram_vocab_size: vec![37, 41, 43],
                max_ngram_size: 4,
                n_embed_per_ngram: 12,
                n_head_per_ngram: 3,
                layer_ids: vec![0, 2],
                pad_id: 0,
                seed: 7,
                kernel_size: 3,
                hc_mult: 1,
                norm_eps: None,
                conv_norm_eps: 1e-5,
                conv_activation: true,
            },
            hidden_size: 8,
        },
    ]
}

fn fixture() -> Result<HashMap<String, Tensor>> {
    candle::safetensors::load(FIXTURE, &Device::Cpu)
}

fn get<'a>(f: &'a HashMap<String, Tensor>, name: &str) -> &'a Tensor {
    f.get(name)
        .unwrap_or_else(|| panic!("missing {name} in the fixture"))
}

fn ids(t: &Tensor) -> Result<Vec<u32>> {
    t.flatten_all()?.to_dtype(DType::U32)?.to_vec1()
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.to_dtype(DType::F32)?;
    let b = b.to_dtype(DType::F32)?;
    (a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar()
}

fn load_modules(
    f: &HashMap<String, Tensor>,
    case: &Case,
    options: &EngramOptions,
) -> Result<(VocabProjection, Vec<Engram>)> {
    let projection = VocabProjection::from_tensor(get(f, &format!("{}.lookup", case.name)))?;
    let hasher = EngramStack::hasher(&case.config, &projection)?;
    let vb = VarBuilder::from_tensors(f.clone(), DType::F32, &Device::Cpu);
    let modules = hasher
        .layers()
        .iter()
        .map(|params| {
            let vb = vb.pp(format!("{}.layer{}.engram", case.name, params.layer_id));
            Engram::new(&case.config, params, case.hidden_size, options, vb)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((projection, modules))
}

#[test]
fn tokenizer_compression_matches_reference() -> Result<()> {
    let words = [
        "<pad>",
        "<s>",
        "</s>",
        " ",
        "  ",
        "\n",
        "\t",
        " \n ",
        "",
        "a",
        "A",
        " a",
        " A",
        "\u{e1}",
        "\u{e4}",
        "\u{c1}",
        "e",
        "E",
        " \u{e9}",
        "\u{e8}",
        "\u{ea}",
        "the",
        "The",
        " the",
        " The",
        "THE",
        " Great",
        " great",
        "Alexander",
        " Alexander",
        "horse",
        " horse",
        " Horse",
        "B",
        "uce",
        "phal",
        "us",
        ".",
        ",",
        "\u{ff27}\u{ff52}\u{ff45}\u{ff41}\u{ff54}",
        "\u{fb01}",
        "fi",
        "\u{fffd}",
        "\u{fffd}\u{fffd}",
        "x\u{fffd}",
        "Stra\u{df}e",
        "strasse",
        "\u{130}stanbul",
        "istanbul",
        "\u{3a9}",
        "\u{3c9}",
    ];
    let vocab: Vec<String> = words
        .iter()
        .map(|w| w.to_string())
        .chain((words.len()..160).map(|i| format!("tok{i}")))
        .chain((0..40).map(|i| format!(" Tok{}", i % 40)))
        .collect();
    let normalizer = reference_normalizer()?;
    let tokens = vocab
        .iter()
        .enumerate()
        .map(|(i, w)| (w.clone(), format!("<raw:{i}>")));
    let projection =
        VocabProjection::from_decoded_vocab(tokens, |s| normalize(&normalizer, s).unwrap());
    let f = fixture()?;
    for case in cases() {
        let expected = ids(get(&f, &format!("{}.lookup", case.name)))?;
        assert_eq!(projection.lookup(), expected.as_slice());
    }
    assert_eq!(projection.vocab_size(), 200);
    assert_eq!(projection.num_compressed(), 174);
    Ok(())
}

#[test]
fn hashing_matches_reference() -> Result<()> {
    let f = fixture()?;
    for case in cases() {
        let projection = VocabProjection::from_tensor(get(&f, &format!("{}.lookup", case.name)))?;
        let hasher = EngramStack::hasher(&case.config, &projection)?;
        let input = get(&f, &format!("{}.input_ids", case.name));
        let (b, t) = input.dims2()?;
        let tokens = projection.project_all(&ids(input)?)?;
        for &layer_id in case.config.layer_ids.iter() {
            let prefix = format!("{}.layer{layer_id}", case.name);
            let params = hasher.layer(layer_id).unwrap();
            let multipliers = get(&f, &format!("{prefix}.multipliers")).to_vec1::<i64>()?;
            assert_eq!(params.multipliers, multipliers, "{prefix}");
            let as_u64 = |t: &Tensor| -> Result<Vec<u64>> {
                Ok(t.flatten_all()?
                    .to_vec1::<i64>()?
                    .into_iter()
                    .map(|v| v as u64)
                    .collect())
            };
            let head_sizes = as_u64(get(&f, &format!("{prefix}.head_sizes")))?;
            assert_eq!(params.head_sizes, head_sizes, "{prefix}");
            let offsets = as_u64(get(
                &f,
                &format!("{prefix}.engram.multi_head_embedding.offsets"),
            ))?;
            assert_eq!(params.head_offsets, offsets, "{prefix}");
            let expected = as_u64(get(&f, &format!("{prefix}.hash_ids")))?;
            let hashes = hasher.hash(layer_id, &tokens, t, &hasher.new_history(b))?;
            assert_eq!(hashes, expected, "{prefix}");
        }
    }
    Ok(())
}

#[test]
fn module_matches_reference() -> Result<()> {
    let f = fixture()?;
    for case in cases() {
        let (projection, modules) = load_modules(&f, &case, &EngramOptions::default())?;
        let hasher = EngramStack::hasher(&case.config, &projection)?;
        let input = get(&f, &format!("{}.input_ids", case.name));
        let (b, t) = input.dims2()?;
        let tokens = projection.project_all(&ids(input)?)?;
        let hidden = get(&f, &format!("{}.hidden_states", case.name));
        for module in modules.iter() {
            let prefix = format!("{}.layer{}", case.name, module.layer_id());
            let expected = get(&f, &format!("{prefix}.output"));
            let rows = hasher.row_ids(module.layer_id(), &tokens, t, &hasher.new_history(b))?;
            let memory = module.retrieve(&rows, b, t)?;
            let (ys, _) = module.forward(hidden, &memory, None)?;
            let diff = max_abs_diff(&ys, expected)?;
            assert!(diff < 1e-5, "{prefix}: {diff}");
            if case.config.hc_mult == 1 {
                // Standard single-stream backbones pass (B, T, D) hidden states.
                let (ys, _) = module.forward(&hidden.squeeze(2)?, &memory, None)?;
                let diff = max_abs_diff(&ys, &expected.squeeze(2)?)?;
                assert!(diff < 1e-5, "{prefix}: {diff}");
            }
        }
    }
    Ok(())
}

#[test]
fn stack_decodes_incrementally() -> Result<()> {
    let f = fixture()?;
    let host = EngramOptions {
        table: TableOptions {
            placement: Placement::Host,
            compression: None,
            prefetch: true,
        },
        ..Default::default()
    };
    for (case, options) in cases()
        .into_iter()
        .flat_map(|c| [(c, EngramOptions::default())])
        .chain(cases().into_iter().map(|c| (c, host)))
    {
        let (projection, modules) = load_modules(&f, &case, &options)?;
        let mut stack = EngramStack::new(case.config.clone(), projection, modules)?;
        let input = get(&f, &format!("{}.input_ids", case.name));
        let hidden = get(&f, &format!("{}.hidden_states", case.name));
        let layer_ids = case.config.layer_ids.clone();

        // The whole sequences at once.
        stack.begin(input, 0)?;
        for &layer_id in layer_ids.iter() {
            let ys = (stack.apply(layer_id, hidden)? - hidden)?;
            let expected = get(&f, &format!("{}.layer{layer_id}.output", case.name));
            let diff = max_abs_diff(&ys, expected)?;
            assert!(diff < 1e-5, "{} layer {layer_id}: {diff}", case.name);
        }
        // Blocks without a module are left untouched.
        assert_eq!(max_abs_diff(&stack.apply(99, hidden)?, hidden)?, 0.);

        // A prompt followed by single tokens and another chunk, as when generating.
        let mut outputs = vec![vec![]; layer_ids.len()];
        for (start, len) in [(0, 5), (5, 1), (6, 1), (7, 6)] {
            stack.begin(&input.narrow(1, start, len)?, start)?;
            let hidden = hidden.narrow(1, start, len)?;
            for (out, &layer_id) in outputs.iter_mut().zip(layer_ids.iter()) {
                out.push((stack.apply(layer_id, &hidden)? - &hidden)?);
            }
        }
        assert_eq!(stack.position(), 13);
        for (out, &layer_id) in outputs.iter().zip(layer_ids.iter()) {
            let ys = Tensor::cat(out, 1)?;
            let expected = get(&f, &format!("{}.layer{layer_id}.output", case.name));
            let diff = max_abs_diff(&ys, expected)?;
            assert!(diff < 1e-5, "{} layer {layer_id}: {diff}", case.name);
        }

        // Misuse is reported rather than silently producing wrong hashes.
        assert!(stack.apply(layer_ids[0], hidden).is_err());
        assert!(stack.begin(&input.narrow(1, 0, 2)?, 3).is_err());
        assert!(stack.begin(&input.narrow(0, 0, 1)?, 13).is_err());
        stack.reset();
        assert!(stack.begin(&input.narrow(1, 0, 2)?, 2).is_err());
    }
    Ok(())
}

#[test]
fn table_storage() -> Result<()> {
    let device = Device::Cpu;
    let table = Tensor::randn(0f32, 1., (300, 64), &device)?;
    let ids = vec![0u32, 299, 17, 17, 42, 128];
    let reference = MemoryTable::on_device(table.clone())?.lookup(&ids)?;
    assert_eq!(reference.dims(), [6, 64]);
    let dense_bytes = 300 * 64 * 4;
    let host = |compression| TableOptions {
        placement: Placement::Host,
        compression,
        prefetch: true,
    };
    let on_device = |compression| TableOptions {
        placement: Placement::Device,
        compression,
        prefetch: false,
    };
    for (options, tolerance, bytes) in [
        (host(None), 0., dense_bytes),
        (host(Some(GgmlDType::F16)), 5e-3, dense_bytes / 2),
        (host(Some(GgmlDType::BF16)), 3e-2, dense_bytes / 2),
        (host(Some(GgmlDType::Q8_0)), 3e-2, 300 * 2 * 34),
        (host(Some(GgmlDType::Q4_0)), 0.6, 300 * 2 * 18),
        (on_device(Some(GgmlDType::F16)), 5e-3, dense_bytes / 2),
        (on_device(Some(GgmlDType::Q8_0)), 3e-2, 300 * 2 * 34),
    ] {
        let t = MemoryTable::from_tensor(table.clone(), &options, &device, DType::F32)?;
        assert_eq!(t.storage_bytes(), bytes, "{options:?}");
        let rows = t.lookup(&ids)?;
        let diff = max_abs_diff(&rows, &reference)?;
        assert!(diff <= tolerance, "{options:?}: {diff}");
        let prefetched = t.prefetch(ids.clone())?.wait()?;
        assert_eq!(max_abs_diff(&rows, &prefetched)?, 0., "{options:?}");
        assert!(t.lookup(&[300]).is_err());
    }

    let dir = std::env::temp_dir().join(format!("candle-engram-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    // Memory-mapped safetensors, in the stored dtype.
    let path = dir.join("tables.safetensors");
    let tensors = HashMap::from([
        (
            "other".to_string(),
            Tensor::ones((3, 5), DType::F32, &device)?,
        ),
        ("table".to_string(), table.clone()),
        ("table_f16".to_string(), table.to_dtype(DType::F16)?),
    ]);
    candle::safetensors::save(&tensors, &path)?;
    let mapped = unsafe { MmapRows::from_safetensors(&path, "table")? };
    let t = MemoryTable::offloaded(Arc::new(mapped), &device, DType::F32);
    assert_eq!(max_abs_diff(&t.lookup(&ids)?, &reference)?, 0.);
    let mapped = unsafe { MmapRows::from_safetensors(&path, "table_f16")? };
    let t = MemoryTable::offloaded(Arc::new(mapped), &device, DType::F32);
    let expected = table.to_dtype(DType::F16)?.to_dtype(DType::F32)?;
    let expected = MemoryTable::on_device(expected)?.lookup(&ids)?;
    assert_eq!(
        max_abs_diff(&t.prefetch(ids.clone())?.wait()?, &expected)?,
        0.
    );
    assert!(unsafe { MmapRows::from_safetensors(&path, "missing") }.is_err());

    // Memory-mapped GGUF, dequantizing only the gathered rows.
    let path = dir.join("tables.gguf");
    let quantized = QTensor::quantize(&table, GgmlDType::Q8_0)?;
    let other = QTensor::quantize(&Tensor::ones((2, 32), DType::F32, &device)?, GgmlDType::F32)?;
    let mut file = std::fs::File::create(&path)?;
    gguf_file::write(&mut file, &[], &[("other", &other), ("table", &quantized)])?;
    drop(file);
    let mapped = unsafe { MmapRows::from_gguf(&path, "table")? };
    let t = MemoryTable::offloaded(Arc::new(mapped), &device, DType::F32);
    let in_memory = HostQuantizedRows::new(Arc::new(quantized))?;
    let expected = MemoryTable::offloaded(Arc::new(in_memory), &device, DType::F32);
    assert_eq!(t.storage_bytes(), 300 * 2 * 34);
    assert_eq!(max_abs_diff(&t.lookup(&ids)?, &expected.lookup(&ids)?)?, 0.);
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}

#[test]
fn zero_init_is_identity_and_trains() -> Result<()> {
    let device = Device::Cpu;
    let config = EngramConfig {
        engram_vocab_size: vec![101, 103],
        n_embed_per_ngram: 16,
        n_head_per_ngram: 2,
        layer_ids: vec![1],
        ..EngramConfig::reference_demo(1)
    };
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let options = EngramOptions {
        zero_init_value: true,
        ..Default::default()
    };
    let projection = VocabProjection::identity(40);
    let mut stack = EngramStack::load(config, projection, 8, &options, vb.pp("layers"))?;
    assert!(stack.modules()[0].table().is_differentiable());

    let input = Tensor::new(
        &[[3u32, 7, 7, 1, 9, 3, 7, 2], [5, 5, 5, 0, 1, 2, 3, 4]],
        &device,
    )?;
    let xs = Tensor::randn(0f32, 1., (2, 8, 8), &device)?;
    stack.begin(&input, 0)?;
    assert_eq!(max_abs_diff(&stack.apply(1, &xs)?, &xs)?, 0.);

    // Fit a target that only depends on the n-grams.
    let target = Tensor::randn(0f32, 1., (2, 8, 8), &device)?;
    let mut opt = candle_nn::AdamW::new_lr(varmap.all_vars(), 3e-2)?;
    let mut losses = vec![];
    for _ in 0..40 {
        stack.begin(&input, 0)?;
        let ys = stack.apply(1, &xs)?;
        let loss = (ys - &target)?.sqr()?.mean_all()?;
        opt.backward_step(&loss)?;
        losses.push(loss.to_scalar::<f32>()?);
    }
    assert!(
        losses[39] < 0.2 * losses[0],
        "the loss did not decrease: {losses:?}"
    );
    Ok(())
}
