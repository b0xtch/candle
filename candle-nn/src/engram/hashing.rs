//! Multi-head n-gram hashing (section 2.2 of the paper).
//!
//! Everything here is plain integer arithmetic on token ids: addresses only depend on the
//! input sequence, never on hidden states. That is what lets the memory tables live away from
//! the accelerator, with rows fetched ahead of the layer that consumes them.
use super::config::EngramConfig;
use super::numpy_rng::Pcg64;
use super::vocab::VocabProjection;
use candle::Result;
use std::collections::HashSet;

fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

fn pow_mod(mut base: u64, mut exp: u64, m: u64) -> u64 {
    let mut acc = 1u64;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = mul_mod(acc, base, m);
        }
        base = mul_mod(base, base, m);
        exp >>= 1;
    }
    acc
}

/// Deterministic Miller-Rabin, exact for every `u64`.
pub fn is_prime(n: u64) -> bool {
    const WITNESSES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    if n < 2 {
        return false;
    }
    for &p in WITNESSES.iter() {
        if n.is_multiple_of(p) {
            return n == p;
        }
    }
    let mut d = n - 1;
    let mut r = 0;
    while d.is_multiple_of(2) {
        d /= 2;
        r += 1;
    }
    'witness: for &a in WITNESSES.iter() {
        let mut x = pow_mod(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..r {
            x = mul_mod(x, x, n);
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

/// Smallest prime strictly greater than `start` that is not in `seen`.
fn next_prime(start: u64, seen: &HashSet<u64>) -> u64 {
    let mut candidate = start + 1;
    while !(is_prime(candidate) && !seen.contains(&candidate)) {
        candidate += 1;
    }
    candidate
}

/// Hashing parameters of one Engram layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerHashParams {
    pub layer_id: usize,
    /// Odd multipliers, one per position in the n-gram window (`max_ngram_size` of them).
    pub multipliers: Vec<i64>,
    /// Prime table size of each head, n-gram order major, head minor.
    pub head_sizes: Vec<u64>,
    /// Offset of each head's table inside the layer's concatenated table.
    pub head_offsets: Vec<u64>,
    /// Total number of rows of the layer's concatenated table.
    pub num_rows: u64,
}

/// Rolling window of the last `max_ngram_size - 1` canonical tokens of every sequence of a
/// batch, so that hashes stay correct when a sequence is fed in several chunks (e.g. prompt
/// processing followed by token-by-token decoding).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NgramHistory {
    window: usize,
    pad_id: u32,
    /// `batch` rows of `window` tokens, oldest first.
    tokens: Vec<Vec<u32>>,
}

impl NgramHistory {
    pub fn new(batch: usize, window: usize, pad_id: u32) -> Self {
        Self {
            window,
            pad_id,
            tokens: vec![vec![pad_id; window]; batch],
        }
    }

    pub fn batch_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn reset(&mut self) {
        for row in self.tokens.iter_mut() {
            row.iter_mut().for_each(|v| *v = self.pad_id)
        }
    }

    /// Appends `seq_len` canonical tokens to every row; `tokens` is `(batch, seq_len)` row-major.
    pub fn push(&mut self, tokens: &[u32], seq_len: usize) {
        for (b, row) in self.tokens.iter_mut().enumerate() {
            let new = &tokens[b * seq_len..(b + 1) * seq_len];
            row.extend_from_slice(new);
            let excess = row.len() - self.window;
            row.drain(..excess);
        }
    }

    fn row(&self, b: usize) -> &[u32] {
        &self.tokens[b]
    }
}

/// Deterministic multi-head n-gram hashing, compatible with `NgramHashMapping` in the
/// reference implementation.
///
/// For a position `t`, the suffix n-gram of canonical tokens `(x_{t-n+1}, ..., x_t)` is mixed
/// as `(x_t * m_0) ^ (x_{t-1} * m_1) ^ ...` with per-layer odd multipliers `m_k`, and each head
/// reduces the mix modulo its own prime table size. Positions before the start of the sequence
/// read the padding token.
#[derive(Debug, Clone)]
pub struct NgramHasher {
    max_ngram_size: usize,
    n_head_per_ngram: usize,
    pad_id: u32,
    layers: Vec<LayerHashParams>,
}

impl NgramHasher {
    /// `compressed_vocab_size` is the size of the canonical vocabulary (it bounds the
    /// multipliers so that products cannot overflow) and `pad_id` the canonical padding id.
    pub fn new(cfg: &EngramConfig, compressed_vocab_size: usize, pad_id: u32) -> Result<Self> {
        cfg.validate()?;
        if compressed_vocab_size == 0 {
            candle::bail!("engram: empty compressed vocabulary")
        }
        let n = cfg.max_ngram_size;
        let m_max = i64::MAX / compressed_vocab_size as i64;
        let half_bound = std::cmp::max(1, m_max / 2);
        const PRIME_1: u64 = 10007;
        let mut seen = HashSet::new();
        let mut layers = Vec::with_capacity(cfg.layer_ids.len());
        for &layer_id in cfg.layer_ids.iter() {
            let seed = cfg.seed.wrapping_add(PRIME_1.wrapping_mul(layer_id as u64));
            let multipliers = Pcg64::default_rng(seed)
                .integers_i64(0, half_bound, n)
                .into_iter()
                .map(|r| r * 2 + 1)
                .collect();
            let mut head_sizes = Vec::with_capacity(cfg.num_heads());
            for &base in cfg.engram_vocab_size.iter() {
                let mut start = base as u64 - 1;
                for _ in 0..cfg.n_head_per_ngram {
                    let p = next_prime(start, &seen);
                    seen.insert(p);
                    head_sizes.push(p);
                    start = p;
                }
            }
            let mut head_offsets = Vec::with_capacity(head_sizes.len());
            let mut num_rows = 0u64;
            for &size in head_sizes.iter() {
                head_offsets.push(num_rows);
                num_rows += size;
            }
            layers.push(LayerHashParams {
                layer_id,
                multipliers,
                head_sizes,
                head_offsets,
                num_rows,
            })
        }
        Ok(Self {
            max_ngram_size: n,
            n_head_per_ngram: cfg.n_head_per_ngram,
            pad_id,
            layers,
        })
    }

    /// The hasher for `cfg` over the canonical ids of `projection`, with the padding id mapped
    /// through it as in the reference implementation.
    pub fn with_projection(cfg: &EngramConfig, projection: &VocabProjection) -> Result<Self> {
        let pad_id = projection.project(cfg.pad_id)?;
        Self::new(cfg, projection.num_compressed(), pad_id)
    }

    pub fn max_ngram_size(&self) -> usize {
        self.max_ngram_size
    }

    /// Number of hash heads per position.
    pub fn num_heads(&self) -> usize {
        (self.max_ngram_size - 1) * self.n_head_per_ngram
    }

    /// Canonical id of the padding token.
    pub fn pad_id(&self) -> u32 {
        self.pad_id
    }

    pub fn layers(&self) -> &[LayerHashParams] {
        &self.layers
    }

    pub fn layer(&self, layer_id: usize) -> Option<&LayerHashParams> {
        self.layers.iter().find(|l| l.layer_id == layer_id)
    }

    /// A history window suitable for this hasher, filled with padding.
    pub fn new_history(&self, batch: usize) -> NgramHistory {
        NgramHistory::new(batch, self.max_ngram_size - 1, self.pad_id)
    }

    fn layer_or_err(&self, layer_id: usize) -> Result<&LayerHashParams> {
        match self.layer(layer_id) {
            Some(l) => Ok(l),
            None => candle::bail!("engram: layer {layer_id} has no hashing parameters"),
        }
    }

    /// Visits the `num_heads()` hashes of every position. `tokens` holds `(batch, seq_len)`
    /// canonical ids, row-major, that follow the tokens recorded in `history`.
    fn for_each_hash<F: FnMut(usize, u64)>(
        &self,
        params: &LayerHashParams,
        tokens: &[u32],
        seq_len: usize,
        history: &NgramHistory,
        mut f: F,
    ) -> Result<()> {
        let batch = history.batch_size();
        if tokens.len() != batch * seq_len {
            candle::bail!(
                "engram: got {} tokens for a batch of {batch} sequences of length {seq_len}",
                tokens.len()
            )
        }
        let n = self.max_ngram_size;
        let k = self.n_head_per_ngram;
        let mut window = vec![0i64; n];
        let mut idx = 0;
        for b in 0..batch {
            let row = &tokens[b * seq_len..(b + 1) * seq_len];
            let past = history.row(b);
            for t in 0..seq_len {
                for (j, w) in window.iter_mut().enumerate() {
                    // Token at position t - j, read from the history when before the chunk.
                    *w = if j <= t {
                        row[t - j] as i64
                    } else {
                        past[past.len() - (j - t)] as i64
                    };
                }
                let mut mix = window[0].wrapping_mul(params.multipliers[0]);
                for order in 2..=n {
                    let j = order - 1;
                    mix ^= window[j].wrapping_mul(params.multipliers[j]);
                    let base = (order - 2) * k;
                    for head in 0..k {
                        let size = params.head_sizes[base + head];
                        // NumPy's `%` is a floor modulo, i.e. `rem_euclid` for a positive size.
                        let hash = mix.rem_euclid(size as i64) as u64;
                        f(idx, hash);
                        idx += 1;
                    }
                }
            }
        }
        Ok(())
    }

    /// Per-head hashes, `(batch, seq_len, num_heads)` row-major, as returned by the reference
    /// `NgramHashMapping.hash` (i.e. without the head offsets).
    pub fn hash(
        &self,
        layer_id: usize,
        tokens: &[u32],
        seq_len: usize,
        history: &NgramHistory,
    ) -> Result<Vec<u64>> {
        let params = self.layer_or_err(layer_id)?;
        let mut out = vec![0u64; tokens.len() * self.num_heads()];
        self.for_each_hash(params, tokens, seq_len, history, |i, h| out[i] = h)?;
        Ok(out)
    }

    /// Rows of the layer's concatenated table to gather, `(batch, seq_len, num_heads)`
    /// row-major: the per-head hashes shifted by the head offsets.
    pub fn row_ids(
        &self,
        layer_id: usize,
        tokens: &[u32],
        seq_len: usize,
        history: &NgramHistory,
    ) -> Result<Vec<u32>> {
        let params = self.layer_or_err(layer_id)?;
        if params.num_rows > u32::MAX as u64 + 1 {
            candle::bail!(
                "engram: layer {layer_id} has {} rows, more than u32 indexing supports",
                params.num_rows
            )
        }
        let heads = self.num_heads();
        let mut out = vec![0u32; tokens.len() * heads];
        self.for_each_hash(params, tokens, seq_len, history, |i, h| {
            out[i] = (h + params.head_offsets[i % heads]) as u32
        })?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primes() {
        let primes: Vec<u64> = (0..60).filter(|&n| is_prime(n)).collect();
        assert_eq!(
            primes,
            [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
        );
        assert!(is_prime(646403));
        assert!(!is_prime(646401));
        assert!(is_prime(18446744073709551557));
        assert!(!is_prime(3215031751)); // strong pseudoprime to bases 2, 3, 5 and 7
        let mut seen = HashSet::new();
        seen.insert(13);
        assert_eq!(next_prime(10, &seen), 11);
        assert_eq!(next_prime(11, &seen), 17);
        assert_eq!(next_prime(17, &seen), 19);
    }

    #[test]
    fn history_chunks() -> Result<()> {
        let cfg = EngramConfig {
            engram_vocab_size: vec![11, 13],
            n_embed_per_ngram: 4,
            n_head_per_ngram: 2,
            layer_ids: vec![0],
            ..EngramConfig::reference_demo(1)
        };
        let hasher = NgramHasher::new(&cfg, 50, 0)?;
        let tokens: Vec<u32> = vec![5, 9, 3, 7, 1, 42, 17, 8, 3, 3, 2, 30];
        let full = hasher.row_ids(0, &tokens, 6, &hasher.new_history(2))?;
        let mut history = hasher.new_history(2);
        let mut chunked = vec![vec![]; 2];
        for (start, len) in [(0, 2), (2, 1), (3, 3)] {
            let chunk: Vec<u32> = (0..2)
                .flat_map(|b| tokens[b * 6 + start..b * 6 + start + len].to_vec())
                .collect();
            let rows = hasher.row_ids(0, &chunk, len, &history)?;
            for (b, out) in chunked.iter_mut().enumerate() {
                out.extend_from_slice(&rows[b * len * 4..(b + 1) * len * 4]);
            }
            history.push(&chunk, len);
        }
        assert_eq!(full, chunked.concat());
        Ok(())
    }
}
