use candle::{DType, Device, Result, Tensor};
use candle_nn::engram::{EngramConfig, EngramOptions, EngramStack, TableOptions, VocabProjection};
use candle_nn::{Activation, VarBuilder, VarMap};
use candle_transformers::models::qwen3;

fn qwen3_config() -> qwen3::Config {
    qwen3::Config {
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 3,
        num_attention_heads: 4,
        head_dim: 8,
        attention_bias: false,
        num_key_value_heads: 2,
        max_position_embeddings: 64,
        sliding_window: None,
        max_window_layers: 3,
        tie_word_embeddings: true,
        rope_theta: 10000.,
        rms_norm_eps: 1e-6,
        use_sliding_window: false,
        hidden_act: Activation::Silu,
    }
}

fn engram_config() -> EngramConfig {
    EngramConfig {
        engram_vocab_size: vec![97, 89],
        n_embed_per_ngram: 16,
        n_head_per_ngram: 2,
        layer_ids: vec![1, 2],
        pad_id: 0,
        ..EngramConfig::reference_demo(1)
    }
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    (a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar()
}

#[test]
fn qwen3_with_engram() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let cfg = qwen3_config();
    let mut model = qwen3::ModelForCausalLM::new(&cfg, vb.clone())?;
    let input = Tensor::new(&[[1u32, 5, 9, 2, 7, 7, 3, 11]], &device)?;
    let base = model.forward(&input, 0)?;
    let projection = VocabProjection::identity(cfg.vocab_size);

    // With zero-initialized value projections, attaching Engram leaves the model unchanged.
    let options = EngramOptions {
        zero_init_value: true,
        ..Default::default()
    };
    let engram = EngramStack::load(
        engram_config(),
        projection.clone(),
        cfg.hidden_size,
        &options,
        vb.pp("engram_zero"),
    )?;
    model.clear_kv_cache();
    model.set_engram(Some(engram));
    assert_eq!(max_abs_diff(&model.forward(&input, 0)?, &base)?, 0.);

    // Otherwise it changes the outputs, and decoding token by token after a prompt matches
    // processing the whole sequence at once.
    let options = EngramOptions {
        table: TableOptions::offloaded(None),
        ..Default::default()
    };
    let engram = EngramStack::load(
        engram_config(),
        projection,
        cfg.hidden_size,
        &options,
        vb.pp("engram"),
    )?;
    model.clear_kv_cache();
    model.set_engram(Some(engram));
    let full = model.forward(&input, 0)?;
    assert!(max_abs_diff(&full, &base)? > 1e-3);
    model.clear_kv_cache();
    model.forward(&input.narrow(1, 0, 5)?, 0)?;
    model.forward(&input.narrow(1, 5, 1)?, 5)?;
    model.forward(&input.narrow(1, 6, 1)?, 6)?;
    let last = model.forward(&input.narrow(1, 7, 1)?, 7)?;
    let diff = max_abs_diff(&last, &full)?;
    assert!(diff < 1e-4, "{diff}");
    assert_eq!(model.engram().map(|e| e.position()), Some(8));
    Ok(())
}
