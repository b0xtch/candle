//! Tokenizer compression (section 2.2 of the paper): n-grams are built from canonical token
//! ids, so that `Apple`, ` apple` and `APPLE` share their n-gram embeddings.
use candle::{DType, Device, Result, Tensor};
use std::collections::HashMap;

/// Surjective projection of raw token ids onto canonical ids ("tokenizer compression").
///
/// Subword vocabularies give distinct ids to strings that are equivalent for n-gram statistics
/// (`Apple`, ` apple`, `APPLE`, ...). Engram hashes n-grams of canonical ids instead, which the
/// paper reports shrinks a 128k vocabulary by 23%.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VocabProjection {
    lookup: Vec<u32>,
    num_ids: usize,
}

impl VocabProjection {
    /// The identity projection, i.e. no tokenizer compression.
    pub fn identity(vocab_size: usize) -> Self {
        Self {
            lookup: (0..vocab_size as u32).collect(),
            num_ids: vocab_size,
        }
    }

    /// Builds a projection from an explicit `raw id -> canonical id` table.
    pub fn from_lookup(lookup: Vec<u32>) -> Result<Self> {
        if lookup.is_empty() {
            candle::bail!("engram: empty vocabulary projection")
        }
        let num_ids = lookup.iter().max().map_or(0, |&m| m as usize + 1);
        Ok(Self { lookup, num_ids })
    }

    /// Builds a projection from one key per raw token id (in id order): tokens with equal keys
    /// share a canonical id, numbered in order of first appearance.
    pub fn from_keys<K, I>(keys: I) -> Self
    where
        K: std::hash::Hash + Eq,
        I: IntoIterator<Item = K>,
    {
        let mut key_to_id = HashMap::new();
        let lookup: Vec<u32> = keys
            .into_iter()
            .map(|key| {
                let next = key_to_id.len() as u32;
                *key_to_id.entry(key).or_insert(next)
            })
            .collect();
        let num_ids = key_to_id.len();
        Self { lookup, num_ids }
    }

    /// Builds the projection the way the reference `CompressedTokenizer` does.
    ///
    /// `tokens` yields, for every raw id in order, the text obtained by decoding that single
    /// id and the raw token string. Decodings containing U+FFFD (partial UTF-8 sequences) are
    /// keyed by their raw token, everything else by `normalize(decoded)`, falling back to the
    /// decoded text when the normalized string is empty.
    pub fn from_decoded_vocab<I, F>(tokens: I, mut normalize: F) -> Self
    where
        I: IntoIterator<Item = (String, String)>,
        F: FnMut(&str) -> String,
    {
        Self::from_keys(tokens.into_iter().map(|(decoded, raw)| {
            if decoded.contains('\u{FFFD}') {
                raw
            } else {
                let normalized = normalize(&decoded);
                if normalized.is_empty() {
                    decoded
                } else {
                    normalized
                }
            }
        }))
    }

    /// The tokenizer compression of the reference implementation for a Hugging Face tokenizer:
    /// every id (added tokens included) is decoded on its own and keyed with
    /// [`reference_normalizer`], see [`VocabProjection::from_decoded_vocab`].
    pub fn from_tokenizer(tokenizer: &tokenizers::Tokenizer) -> Result<Self> {
        let normalizer = reference_normalizer()?;
        let vocab_size = tokenizer.get_vocab_size(true) as u32;
        let tokens = (0..vocab_size)
            .map(|id| {
                let decoded = tokenizer
                    .decode(&[id], false)
                    .map_err(candle::Error::wrap)?;
                let raw = tokenizer.id_to_token(id).unwrap_or_default();
                Ok((decoded, raw))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut error = None;
        let projection = Self::from_decoded_vocab(tokens, |s| {
            normalize(&normalizer, s).unwrap_or_else(|e| {
                error.get_or_insert(e);
                String::new()
            })
        });
        match error {
            Some(e) => Err(e),
            None => Ok(projection),
        }
    }

    /// Loads a projection stored as an integer tensor of shape `(vocab_size,)`.
    pub fn from_tensor(t: &Tensor) -> Result<Self> {
        let lookup = t.flatten_all()?.to_dtype(DType::U32)?.to_vec1::<u32>()?;
        Self::from_lookup(lookup)
    }

    /// The lookup table as a `u32` tensor, e.g. to store it alongside the Engram weights.
    pub fn to_tensor(&self, device: &Device) -> Result<Tensor> {
        Tensor::new(self.lookup.as_slice(), device)
    }

    pub fn lookup(&self) -> &[u32] {
        &self.lookup
    }

    /// Number of raw token ids covered by the projection.
    pub fn vocab_size(&self) -> usize {
        self.lookup.len()
    }

    /// Size of the canonical vocabulary.
    pub fn num_compressed(&self) -> usize {
        self.num_ids
    }

    /// Fraction of the raw vocabulary removed by the projection.
    pub fn compression_ratio(&self) -> f64 {
        1.0 - self.num_ids as f64 / self.lookup.len() as f64
    }

    pub fn project(&self, id: u32) -> Result<u32> {
        match self.lookup.get(id as usize) {
            Some(&v) => Ok(v),
            None => candle::bail!(
                "engram: token id {id} is outside the vocabulary projection ({} ids)",
                self.lookup.len()
            ),
        }
    }

    pub fn project_all(&self, ids: &[u32]) -> Result<Vec<u32>> {
        ids.iter().map(|&id| self.project(id)).collect()
    }
}

/// The normalizer of the reference `CompressedTokenizer`: NFKC, NFD, accent stripping,
/// lowercasing, collapsing runs of whitespace into a single space and stripping, except that a
/// lone space is preserved.
pub fn reference_normalizer() -> Result<tokenizers::NormalizerWrapper> {
    use tokenizers::normalizers::replace::ReplacePattern;
    use tokenizers::normalizers::{Lowercase, Replace, Sequence, Strip, StripAccents, NFD, NFKC};
    const SENTINEL: &str = "\u{E000}";
    let replace = |pattern: ReplacePattern, content: &str| {
        Replace::new(pattern, content).map_err(candle::Error::wrap)
    };
    let regex = |r: &str| ReplacePattern::Regex(r.to_string());
    let string = |s: &str| ReplacePattern::String(s.to_string());
    Ok(Sequence::new(vec![
        NFKC.into(),
        NFD.into(),
        StripAccents.into(),
        Lowercase.into(),
        replace(regex(r"[ \t\r\n]+"), " ")?.into(),
        replace(regex(r"^ $"), SENTINEL)?.into(),
        Strip::new(true, true).into(),
        replace(string(SENTINEL), " ")?.into(),
    ])
    .into())
}

/// Applies a tokenizer normalizer to a string.
pub fn normalize(normalizer: &impl tokenizers::Normalizer, s: &str) -> Result<String> {
    let mut normalized = tokenizers::NormalizedString::from(s);
    normalizer
        .normalize(&mut normalized)
        .map_err(candle::Error::wrap)?;
    Ok(normalized.get().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection() {
        let p = VocabProjection::from_keys(["a", "b", "a", "c", "b"]);
        assert_eq!(p.lookup(), [0, 1, 0, 2, 1]);
        assert_eq!(p.num_compressed(), 3);
        assert_eq!(p.vocab_size(), 5);
        assert!(p.project(5).is_err());
        let tokens = [
            ("Hello".to_string(), "Hello".to_string()),
            (" hello".to_string(), "Ġhello".to_string()),
            ("\u{FFFD}".to_string(), "<0xE2>".to_string()),
            ("\u{FFFD}".to_string(), "<0x82>".to_string()),
            ("   ".to_string(), "ĠĠĠ".to_string()),
        ];
        let p = VocabProjection::from_decoded_vocab(tokens, |s| s.trim().to_lowercase());
        // Partial UTF-8 decodings keep distinct ids, empty normalizations keep the raw text.
        assert_eq!(p.lookup(), [0, 0, 1, 2, 3]);
    }

    #[test]
    fn normalizer() -> Result<()> {
        let n = reference_normalizer()?;
        let cases = [
            (" The", "the"),
            ("  Ｇｒｅａｔ\t\n", "great"),
            ("Á", "a"),
            (" ", " "),
            ("\n", " "),
            (" \n ", " "),
            ("", ""),
            ("ﬁ", "fi"),
        ];
        for (input, expected) in cases {
            assert_eq!(normalize(&n, input)?, expected, "{input:?}");
        }
        Ok(())
    }
}
