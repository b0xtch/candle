# candle-engram

[Engram](https://github.com/deepseek-ai/Engram) ("Conditional Memory via Scalable Lookup",
DeepSeek-AI 2026) gives a transformer a large table of hashed n-gram embeddings. At a few decoder
blocks, the embeddings of the n-grams ending at each position are looked up in O(1), gated by the
hidden state and added to the residual stream. As the addresses only depend on the token ids, the
tables can live in host memory or on disk, with the rows a forward pass needs fetched ahead of
time.

This example attaches Engram, from `candle_nn::engram`, to a quantized Qwen3 model and shows the
storage options:

- `--storage device|host|mmap`: tables on the compute device, in host memory with only the
  looked up rows copied to the device, or memory mapped from a GGUF file and paged in on demand.
- `--compression none|f16|bf16|q8_0|q4_0`: how the tables are encoded.
- `--no-prefetch`: gather offloaded rows when the Engram layer runs rather than on a background
  thread as soon as the token ids are known.

It also reports the tokenizer compression of the model's vocabulary, i.e. how many raw token ids
collapse onto the same canonical id once case, accents and whitespace are folded.

## Running the example

Without trained Engram weights, fresh modules are created with zero value projections, so that
the model outputs are unchanged and the run measures the cost of the lookups:

```bash
cargo run --example engram --release -- --storage host --compression q8_0 --compare
```

The run prints the size and location of every table, the prompt and generation speed with and
without Engram, and checks that both runs produce the same tokens.

Tables larger than memory can be memory mapped; the file is written on the first run:

```bash
cargo run --example engram --release -- --storage mmap --compression q4_0 --tables-file engram.gguf
```

Trained modules are loaded with `--engram-weights model.safetensors --engram-config engram.json`,
where the weights use the names of the reference implementation under `layers.{i}.engram`
(`multi_head_embedding.embedding.weight`, `value_proj`, `key_projs.{m}`, `norm1.{m}`, `norm2.{m}`,
`short_conv.conv.weight`, `short_conv.norms.{m}`) and the config its `EngramConfig` fields, e.g.

```json
{
  "engram_vocab_size": [646400, 646400],
  "max_ngram_size": 3,
  "n_embed_per_ngram": 512,
  "n_head_per_ngram": 8,
  "layer_ids": [1, 15],
  "pad_id": 151643,
  "seed": 0,
  "kernel_size": 4
}
```

## Using Engram in other models

`EngramStack` holds everything a decoder needs: the tokenizer compression, the hashing, the modules
and the per-sequence state (the last tokens of each sequence and the convolution states). A model
only needs two calls per forward pass, as in `candle_transformers::models::qwen3`:

```rust
if let Some(engram) = self.engram.as_mut() {
    // Hash the n-grams of the new tokens and start fetching their memory.
    engram.begin(input_ids, seqlen_offset)?;
}
let mut xs = self.embed_tokens.forward(input_ids)?;
for (i, layer) in self.layers.iter_mut().enumerate() {
    if let Some(engram) = self.engram.as_mut() {
        // H <- H + Engram(H) for the blocks that have a module.
        xs = engram.apply(i, &xs)?;
    }
    xs = layer.forward(&xs, mask, seqlen_offset)?;
}
```

and `engram.reset()` wherever the KV cache is cleared.
