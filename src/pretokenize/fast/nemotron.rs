//! Fast pretokenizer for the Nemotron-3 regex (nvidia Nemotron-3 family):
//! `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`
//!
//! The o200k scheme without contraction suffixes and with single-char
//! `\p{N}` digit tokens. See `o200k_family` (`CONTRACTIONS = false`,
//! `DIGITS3 = false`).

use super::mask::{MaskScheme, MaskState};
use super::o200k_family;
use super::{decode_cp, is_digit};
use crate::pretokenize::Pretoken;
use crate::pretokenize::unicode::{O200kCharClass, o200k_class_of};

pub(crate) struct NemotronScheme;

impl NemotronScheme {
    #[inline(always)]
    fn class_of(cp: u32) -> O200kCharClass {
        o200k_class_of(cp)
    }
}

impl MaskScheme for NemotronScheme {
    #[inline(always)]
    fn advance(bytes: &[u8], pos: usize) -> usize {
        o200k_family::advance_pos::<false, false, true, false>(bytes, pos)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn batch_masks(bytes: &[u8], scan: usize) -> (u64, u64) {
        o200k_family::batch_masks::<false, false, true, false>(bytes, scan)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn batch_masks_x86<const AVX512: bool>(bytes: &[u8], scan: usize) -> (u64, u64) {
        // SAFETY: the caller detected the tier (trait contract).
        unsafe { o200k_family::batch_masks_x86::<AVX512, false, false, true, false>(bytes, scan) }
    }
}

/// With SIMD support (aarch64 NEON, or x86_64 AVX-512/AVX2 detected at
/// runtime), iteration runs the shared o200k-family mask scanner (see
/// `o200k_family::batch_masks`); elsewhere every token takes the scalar
/// `advance_pos`.
pub struct FastNemotronPretokenizer<'a> {
    bytes: &'a [u8],
    state: MaskState,
}

impl<'a> FastNemotronPretokenizer<'a> {
    #[inline]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self::with_pos(bytes, 0)
    }

    /// Resume iteration at a byte offset previously returned by [`Self::pos`].
    #[inline]
    pub fn with_pos(bytes: &'a [u8], pos: usize) -> Self {
        Self { bytes, state: MaskState::new(pos) }
    }

    /// Current position as a byte offset into the input.
    #[inline]
    pub fn pos(&self) -> usize {
        self.state.pos
    }
}

impl<'a> Iterator for FastNemotronPretokenizer<'a> {
    type Item = Pretoken<'a>;

    #[inline]
    fn next(&mut self) -> Option<Pretoken<'a>> {
        let (start, end) = self.state.next_span::<NemotronScheme>(self.bytes)?;
        Some(Pretoken(&self.bytes[start..end]))
    }
}

super::impl_mask_pretoken_spans!(FastNemotronPretokenizer, NemotronScheme);

/// Right-aligned Unicode Number groups of up to three; all other spans use
/// the existing Nemotron scanner. Only this iterator carries grouping state.
pub struct FastNemotronRightAlignedPretokenizer<'a> {
    bytes: &'a [u8],
    state: MaskState,
    numbers_left: usize,
}

impl<'a> FastNemotronRightAlignedPretokenizer<'a> {
    #[inline]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self::with_pos(bytes, 0)
    }

    /// Resume at a byte offset previously returned by [`Self::pos`].
    #[inline]
    pub fn with_pos(bytes: &'a [u8], pos: usize) -> Self {
        Self {
            bytes,
            state: MaskState::new(pos),
            numbers_left: 0,
        }
    }

    #[inline]
    pub fn pos(&self) -> usize {
        self.state.pos
    }

    #[inline(always)]
    fn number_end(&self, pos: usize) -> Option<usize> {
        let &b = self.bytes.get(pos)?;
        if b < 0x80 {
            return is_digit(b).then_some(pos + 1);
        }
        // SAFETY: pos < len and b is non-ASCII; decode_cp checks the tail.
        let (cp, width) = unsafe { decode_cp(self.bytes, pos) };
        (NemotronScheme::class_of(cp) == O200kCharClass::Number).then_some(pos + width)
    }

    #[inline(always)]
    fn next_span(&mut self) -> Option<(usize, usize)> {
        let start;
        if self.numbers_left == 0 {
            let span = self.state.next_span::<NemotronScheme>(self.bytes)?;
            let Some(mut end) = self.number_end(span.0) else {
                return Some(span);
            };
            start = span.0;
            self.numbers_left = 1;
            // Count the complete run once. The first group is the remainder;
            // the remaining groups each contain three Unicode codepoints.
            while let Some(next) = self.number_end(end) {
                end = next;
                self.numbers_left += 1;
            }
        } else {
            start = self.state.pos;
        }
        let count = (self.numbers_left - 1) % 3 + 1;
        let mut end = start;
        for _ in 0..count {
            end = self.number_end(end).expect("counted number run");
        }
        self.numbers_left -= count;
        // Discard single-number boundaries cached inside this run.
        self.state = MaskState::new(end);
        Some((start, end))
    }
}

impl<'a> Iterator for FastNemotronRightAlignedPretokenizer<'a> {
    type Item = Pretoken<'a>;

    #[inline]
    fn next(&mut self) -> Option<Pretoken<'a>> {
        let (start, end) = self.next_span()?;
        Some(Pretoken(&self.bytes[start..end]))
    }
}

// SAFETY: next_span emits only live, nonempty spans within self.bytes.
unsafe impl<'a> crate::pretokenize::PretokenSpans<'a> for FastNemotronRightAlignedPretokenizer<'a> {
    #[inline]
    fn fill_spans_keyed(
        &mut self,
        batch: &mut crate::pretokenize::SpanBatch<'a>,
        prefetch: &impl Fn(u64),
    ) -> usize {
        let bytes = self.bytes;
        crate::pretokenize::fill_spans_keyed_with_buf(bytes, || self.next_span(), batch, prefetch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pretokenize::{PretokenSpans, SpanBatch, safe_split_ranges};

    /// The Nemotron pattern verbatim — no possessive quantifiers, so it
    /// runs directly under fancy-regex.
    const NEMOTRON_REF_REGEX: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const NEMOTRON_RIGHT_ALIGNED_REF_REGEX: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}{1,3}(?=(?:\p{N}{3})*(?!\p{N}))|\p{N}{1,2}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    fn regex_tokens(s: &str) -> Vec<String> {
        let re = fancy_regex::Regex::new(NEMOTRON_REF_REGEX).unwrap();
        re.find_iter(s)
            .map(|m| m.unwrap().as_str().to_string())
            .collect()
    }

    fn fast_tokens(s: &str) -> Vec<String> {
        FastNemotronPretokenizer::new(s.as_bytes())
            .map(|t| String::from_utf8_lossy(t.0).into_owned())
            .collect()
    }

    /// The o200k small-case list applies verbatim (contraction cases just
    /// tokenize differently, which the reference regex reflects).
    #[test]
    fn nemotron_small_cases() {
        for case in crate::pretokenize::fast::o200k::tests::SMALL_CASES {
            assert_eq!(
                fast_tokens(case),
                regex_tokens(case),
                "Mismatch on case {case:?}"
            );
        }
    }

    /// Random codepoint soup drawn from classes the scheme distinguishes,
    /// compared against the reference regex.
    #[test]
    fn nemotron_matches_regex_random() {
        use rand::prelude::*;
        let pools: &[&[char]] = &[
            &['a', 'z', 'é', 'ß', 'ж', 'ا', '한', '日'],      // lower/caseless
            &['A', 'Z', 'É', 'Ж', 'Ǆ', 'ǅ'],                  // upper/title
            &['1', '9', '٢', '½', 'Ⅷ', '๕'],                // numbers
            &[' ', '\t', '\n', '\r', '\u{a0}', '\u{2028}'],   // whitespace
            &['\u{301}', '\u{5bf}', '\u{93b}', '\u{20dd}'],   // marks
            &['.', ',', '!', '$', '\'', '«', '¡', '€', '☃', '/'], // punct/symbols
            &['\u{0}', '\u{ad}', '\u{200b}', '\u{e0001}'],    // other (C*)
        ];
        let mut rng = StdRng::seed_from_u64(0x93E3_5EEE);
        for round in 0..3000 {
            let len = rng.random_range(1..40);
            let s: String = (0..len)
                .map(|_| {
                    let pool = pools.choose(&mut rng).unwrap();
                    *pool.choose(&mut rng).unwrap()
                })
                .collect();
            assert_eq!(
                fast_tokens(&s),
                regex_tokens(&s),
                "Mismatch on round {round}, case {s:?}"
            );
        }
    }


    fn assert_right_aligned_reference(text: &str) {
        static RE: std::sync::OnceLock<fancy_regex::Regex> = std::sync::OnceLock::new();
        let re = RE.get_or_init(|| fancy_regex::Regex::new(NEMOTRON_RIGHT_ALIGNED_REF_REGEX).unwrap());
        let expected: Vec<&[u8]> = re
            .find_iter(text)
            .map(|m| m.unwrap().as_str().as_bytes())
            .collect();
        let actual: Vec<&[u8]> = FastNemotronRightAlignedPretokenizer::new(text.as_bytes())
            .map(|p| p.0)
            .collect();
        assert_eq!(actual, expected, "{text:?}");
        let mut scanner = FastNemotronRightAlignedPretokenizer::new(text.as_bytes());
        let mut batch = SpanBatch::new();
        let mut filled = Vec::new();
        loop {
            let n = scanner.fill_spans_keyed(&mut batch, &|_| {});
            if n == 0 {
                break;
            }
            for i in 0..n {
                // SAFETY: i was initialized by the immediately preceding fill.
                filled.push(unsafe { batch.span(i) });
            }
        }
        assert_eq!(filled, expected);
    }

    #[test]
    fn right_aligned_numbers_and_vector_edges() {
        for n in 1..130 {
            for prefix in 0..70 {
                assert_right_aligned_reference(&format!("{} {} suffix", "a".repeat(prefix), "1".repeat(n)));
            }
        }
        for text in [
            "١٢٣٤٥٦٧",
            "１２３４５",
            "ⅧⅡⅣⅦⅤ",
            "½¼¾⅓⅔",
            "1٢３Ⅷ½4٥6",
            "cafe\u{301}A 1234\r\n//abc",
        ] {
            assert_right_aligned_reference(text);
        }
    }

    #[test]
    fn right_aligned_random_reference() {
        use rand::prelude::*;
        let pool: Vec<char> = "aAZz019 \r\n\t.,/'éßЖا한日١２Ⅷ½\u{301}🌍".chars().collect();
        let mut rng = StdRng::seed_from_u64(420);
        for _ in 0..3000 {
            let n = rng.random_range(1..120);
            let text: String = (0..n).map(|_| *pool.choose(&mut rng).unwrap()).collect();
            assert_right_aligned_reference(&text);
        }
    }

    #[test]
    fn right_aligned_long_run_and_safe_cuts() {
        let text = "1".repeat(100_001);
        let groups: Vec<_> = FastNemotronRightAlignedPretokenizer::new(text.as_bytes())
            .map(|p| p.0)
            .collect();
        assert_eq!(groups[0].len(), 2);
        assert!(groups[1..].iter().all(|p| p.len() == 3));
        assert_eq!(groups.concat(), text.as_bytes());
        let text = "alpha 1234567 beta ١٢٣٤٥٦٧ gamma\n".repeat(1000);
        let whole: Vec<_> = FastNemotronRightAlignedPretokenizer::new(text.as_bytes())
            .map(|p| p.0)
            .collect();
        let ranges = safe_split_ranges(text.as_bytes(), 100, &[]);
        assert!(ranges.len() > 1);
        let fragments: Vec<_> = ranges
            .iter()
            .flat_map(|r| {
                FastNemotronRightAlignedPretokenizer::new(&text.as_bytes()[r.clone()]).map(|p| p.0)
            })
            .collect();
        assert_eq!(whole, fragments);
    }
}
