//! A minimal port of NumPy's default random generator.
//!
//! The reference Engram implementation derives its hash multipliers with
//! `np.random.default_rng(seed).integers(0, high, size, dtype=np.int64)`. Reproducing that
//! stream exactly is what makes the n-gram hashes (and therefore checkpoints trained with the
//! reference code) portable to candle, so this module implements the pieces involved:
//! `SeedSequence` entropy mixing, the `PCG64` bit generator (XSL-RR output) and the
//! Lemire bounded-integer sampler used by `Generator.integers`.

const POOL_SIZE: usize = 4;
const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;

const PCG_MULTIPLIER: u128 = (2549297995355413924u128 << 64) | 4865540595714422341u128;

fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut value = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    value = value.wrapping_mul(*hash_const);
    value ^ (value >> XSHIFT)
}

fn mix(x: u32, y: u32) -> u32 {
    let result = MIX_MULT_L
        .wrapping_mul(x)
        .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    result ^ (result >> XSHIFT)
}

/// NumPy's `SeedSequence` restricted to a single integer of entropy and no spawn key.
#[derive(Debug, Clone)]
pub struct SeedSequence {
    pool: [u32; POOL_SIZE],
}

impl SeedSequence {
    pub fn new(seed: u64) -> Self {
        // `_int_to_uint32_array`: little-endian 32-bit words, `[0]` for a zero seed.
        let mut entropy = vec![];
        let mut n = seed;
        if n == 0 {
            entropy.push(0u32);
        }
        while n > 0 {
            entropy.push((n & 0xffff_ffff) as u32);
            n >>= 32;
        }
        let mut pool = [0u32; POOL_SIZE];
        let mut hash_const = INIT_A;
        for (i, p) in pool.iter_mut().enumerate() {
            let v = entropy.get(i).copied().unwrap_or(0);
            *p = hashmix(v, &mut hash_const);
        }
        for i_src in 0..POOL_SIZE {
            for i_dst in 0..POOL_SIZE {
                if i_src != i_dst {
                    let h = hashmix(pool[i_src], &mut hash_const);
                    pool[i_dst] = mix(pool[i_dst], h);
                }
            }
        }
        for &e in entropy.iter().skip(POOL_SIZE) {
            for p in pool.iter_mut() {
                let h = hashmix(e, &mut hash_const);
                *p = mix(*p, h);
            }
        }
        Self { pool }
    }

    /// `generate_state(n_words, np.uint64)`.
    pub fn generate_state_u64(&self, n_words: usize) -> Vec<u64> {
        let mut hash_const = INIT_B;
        let words: Vec<u32> = (0..2 * n_words)
            .map(|i| {
                let mut v = self.pool[i % POOL_SIZE] ^ hash_const;
                hash_const = hash_const.wrapping_mul(MULT_B);
                v = v.wrapping_mul(hash_const);
                v ^ (v >> XSHIFT)
            })
            .collect();
        words
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&[lo, hi]| (lo as u64) | ((hi as u64) << 32))
            .collect()
    }
}

/// NumPy's `PCG64` bit generator, including the buffered 32-bit output.
#[derive(Debug, Clone)]
pub struct Pcg64 {
    state: u128,
    inc: u128,
    buffered_u32: Option<u32>,
}

impl Pcg64 {
    pub fn from_seed_sequence(seq: &SeedSequence) -> Self {
        let v = seq.generate_state_u64(4);
        let init_state = ((v[0] as u128) << 64) | v[1] as u128;
        let init_seq = ((v[2] as u128) << 64) | v[3] as u128;
        let mut rng = Self {
            state: 0,
            inc: (init_seq << 1) | 1,
            buffered_u32: None,
        };
        rng.step();
        rng.state = rng.state.wrapping_add(init_state);
        rng.step();
        rng
    }

    /// The generator returned by `np.random.default_rng(seed)`.
    pub fn default_rng(seed: u64) -> Self {
        Self::from_seed_sequence(&SeedSequence::new(seed))
    }

    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(PCG_MULTIPLIER)
            .wrapping_add(self.inc);
    }

    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let state = self.state;
        let rot = (state >> 122) as u32;
        (((state >> 64) as u64) ^ (state as u64)).rotate_right(rot)
    }

    pub fn next_u32(&mut self) -> u32 {
        if let Some(v) = self.buffered_u32.take() {
            return v;
        }
        let next = self.next_u64();
        self.buffered_u32 = Some((next >> 32) as u32);
        (next & 0xffff_ffff) as u32
    }

    fn bounded_lemire_u64(&mut self, rng: u64) -> u64 {
        let rng_excl = rng + 1;
        let mut m = (self.next_u64() as u128) * rng_excl as u128;
        let mut leftover = m as u64;
        if leftover < rng_excl {
            let threshold = (u64::MAX - rng) % rng_excl;
            while leftover < threshold {
                m = (self.next_u64() as u128) * rng_excl as u128;
                leftover = m as u64;
            }
        }
        (m >> 64) as u64
    }

    fn bounded_lemire_u32(&mut self, rng: u32) -> u32 {
        let rng_excl = rng + 1;
        let mut m = (self.next_u32() as u64) * rng_excl as u64;
        let mut leftover = m as u32;
        if leftover < rng_excl {
            let threshold = (u32::MAX - rng) % rng_excl;
            while leftover < threshold {
                m = (self.next_u32() as u64) * rng_excl as u64;
                leftover = m as u32;
            }
        }
        (m >> 32) as u32
    }

    /// `Generator.integers(low, high, size, dtype=np.int64)` with the default `endpoint=False`.
    ///
    /// Requires `low < high`.
    pub fn integers_i64(&mut self, low: i64, high: i64, size: usize) -> Vec<i64> {
        assert!(
            low < high,
            "integers requires low < high, got {low} >= {high}"
        );
        // NumPy samples on the closed interval [low, high - 1].
        let rng = (high - 1).wrapping_sub(low) as u64;
        let off = low as u64;
        let mut out = Vec::with_capacity(size);
        if rng == 0 {
            out.resize(size, low);
        } else if rng <= 0xffff_ffff {
            for _ in 0..size {
                let v = if rng == 0xffff_ffff {
                    self.next_u32() as u64
                } else {
                    self.bounded_lemire_u32(rng as u32) as u64
                };
                out.push(off.wrapping_add(v) as i64)
            }
        } else if rng == u64::MAX {
            for _ in 0..size {
                out.push(off.wrapping_add(self.next_u64()) as i64)
            }
        } else {
            for _ in 0..size {
                out.push(off.wrapping_add(self.bounded_lemire_u64(rng)) as i64)
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Values produced by NumPy 2.4:
    //   [int(v) for v in np.random.default_rng(seed).integers(0, high, size=(3,), dtype=np.int64)]
    #[test]
    fn matches_numpy_integers_64bit() {
        let high = 46116860184273;
        let cases: [(u64, [i64; 3]); 4] = [
            (0, [29374673076942, 12441716158222, 1889570274622]),
            (1, [23603606305589, 43832401393690, 6648188704010]),
            (10007, [37968138307072, 2397965009193, 17816579247344]),
            (
                (1 << 40) + 3,
                [35678589505950, 45925815566218, 21821525846935],
            ),
        ];
        for (seed, expected) in cases {
            let got = Pcg64::default_rng(seed).integers_i64(0, high, 3);
            assert_eq!(got, expected, "seed {seed}");
        }
    }

    #[test]
    fn matches_numpy_raw_and_32bit_paths() {
        // np.random.default_rng(5).bit_generator.random_raw(3)
        let mut rng = Pcg64::default_rng(5);
        let raw: Vec<u64> = (0..3).map(|_| rng.next_u64()).collect();
        assert_eq!(
            raw,
            [
                14849682912918955432,
                14903876974979881461,
                9506078739185184192
            ]
        );
        // Ranges that fit in 32 bits go through the buffered 32-bit Lemire sampler.
        let got = Pcg64::default_rng(5).integers_i64(0, 1000, 7);
        assert_eq!(got, [670, 805, 22, 807, 468, 515, 630]);
        let got = Pcg64::default_rng(6).integers_i64(0, 1 << 32, 5);
        assert_eq!(
            got,
            [1911456510, 2311398289, 2223771871, 1474337159, 4063638511]
        );
        let got = Pcg64::default_rng(7).integers_i64(-5, 7, 6);
        assert_eq!(got, [6, 2, 3, 5, 1, 4]);
        let got = Pcg64::default_rng(9).integers_i64(4, 5, 3);
        assert_eq!(got, [4, 4, 4]);
    }
}
