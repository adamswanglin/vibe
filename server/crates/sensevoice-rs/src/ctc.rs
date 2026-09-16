//! Greedy CTC decoding and SentencePiece detokenization, matching the `funasr-sensevoice` decode
//! loop: per frame take the argmax, drop repeats, drop the blank, then join the pieces.

/// Collapses `frames` rows of `vocab` logits into text.
pub(crate) fn decode(scores: &[f32], frames: usize, vocab: usize, blank: usize, pieces: &[String]) -> String {
    let mut ids = Vec::new();
    let mut previous = usize::MAX;
    for frame in 0..frames {
        let column = &scores[frame * vocab..(frame + 1) * vocab];
        let mut best = 0;
        for (index, value) in column.iter().enumerate().skip(1) {
            if *value > column[best] {
                best = index;
            }
        }
        if best != previous && best != blank {
            ids.push(best);
        }
        previous = best;
    }
    detokenize(&ids, pieces)
}

/// Joins the pieces, drops the `<|lang|>`-style meta tokens, and turns SentencePiece's word marker
/// (U+2581) back into spaces.
pub(crate) fn detokenize(ids: &[usize], pieces: &[String]) -> String {
    let mut text = String::new();
    for id in ids {
        let Some(piece) = pieces.get(*id) else {
            continue;
        };
        if piece.starts_with("<|") {
            continue;
        }
        text.push_str(piece);
    }
    text.replace('\u{2581}', " ").trim_matches(' ').to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces() -> Vec<String> {
        ["<blank>", "<|zh|>", "\u{2581}he", "llo", "\u{2581}world"]
            .iter()
            .map(|piece| (*piece).to_owned())
            .collect()
    }

    #[test]
    fn collapses_repeats_and_drops_the_blank() {
        // Rows: 2 2 0 3 1 4 4 -> "he" "llo" "<|zh|>" "world"
        let rows = [2usize, 2, 0, 3, 1, 4, 4];
        let mut scores = vec![0.0f32; rows.len() * pieces().len()];
        for (frame, id) in rows.iter().enumerate() {
            scores[frame * pieces().len() + id] = 1.0;
        }
        assert_eq!(decode(&scores, rows.len(), pieces().len(), 0, &pieces()), "hello world");
    }

    #[test]
    fn blank_only_audio_decodes_to_nothing() {
        let scores = vec![1.0f32; 5];
        assert_eq!(decode(&scores, 1, 5, 0, &pieces()), "");
    }

    #[test]
    fn unknown_ids_and_leading_spaces_are_dropped() {
        assert_eq!(detokenize(&[9, 2, 3], &pieces()), "hello");
        assert_eq!(detokenize(&[3, 2], &pieces()), "llo he");
    }
}
