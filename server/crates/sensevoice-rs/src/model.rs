//! The loaded model and what it can do: run one chunk through the encoder, and walk a long
//! recording segment by segment over the VAD ranges.
//!
//! Loading a GGUF into one of these — metadata, vocabulary, tensor catalog, GPU upload — lives in
//! [`crate::load`].

use std::sync::Mutex;
use std::time::Instant;

use crate::ctc;
use crate::encoder;
use crate::fbank;
use crate::load::ModelInfo;
use crate::runtime::{Runtime, Weights};
use crate::{sys, Error, Result};

/// How to pick the inference device. `no_gpu` forces the CPU; otherwise `gpu_device` addresses the
/// n-th GPU or IGPU device, with -1 meaning the first one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadOptions {
    pub no_gpu: bool,
    pub gpu_device: i32,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            no_gpu: false,
            gpu_device: -1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    /// Milliseconds from the start of the audio.
    pub start: i64,
    pub end: i64,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Transcription {
    pub segments: Vec<Segment>,
}

/// Loaded GGUF metadata, the vocabulary, the query-token rows, and the tensor storage.
///
/// The weights buffer must outlive every graph run, so it is kept beside the runtime that owns the
/// backends it was allocated from — and declared first, so `Drop` frees the buffer before the
/// backends go away.
pub struct Model {
    pub(crate) weights: Weights,
    pub(crate) info: ModelInfo,
    pub(crate) vocab: Vec<String>,
    /// The `embed.weight` rows the `sv.query_tokens` prompt names, copied out at load time in
    /// token order.
    pub(crate) embeddings: Vec<f32>,
    pub(crate) runtime: Mutex<Runtime>,
}

unsafe impl Send for Model {}

impl Model {
    pub fn info(&self) -> &ModelInfo {
        &self.info
    }

    pub(crate) unsafe fn tensor(&self, name: &str) -> Option<*mut sys::ggml_tensor> {
        self.weights.tensor(name)
    }

    pub fn transcribe(&self, vad: &mut vad_rs::Vad, samples: &[f32]) -> Result<Transcription> {
        self.transcribe_with(vad, samples, || false, |_| {}, |_| {})
    }

    /// Splits the audio on the VAD ranges and transcribes each one, reporting progress on the
    /// share of audio already covered.
    pub fn transcribe_with(
        &self,
        vad: &mut vad_rs::Vad,
        samples: &[f32],
        mut should_abort: impl FnMut() -> bool,
        mut on_segment: impl FnMut(&Segment),
        mut on_progress: impl FnMut(i32),
    ) -> Result<Transcription> {
        let started = Instant::now();
        let ranges = vad.segments(samples).map_err(|error| Error::Vad(error.to_string()))?;
        let mut segments = Vec::new();
        for range in ranges {
            if should_abort() {
                return Err(Error::Aborted);
            }
            let end = range.end_sample.min(samples.len());
            let start = range.start_sample.min(end);
            let chunk = &samples[start..end];
            // VAD happily reports ranges shorter than one analysis window or pure zeros.
            if chunk.len() >= fbank::WIN_LEN && chunk.iter().any(|&sample| sample != 0.0) {
                let text = self.transcribe_chunk(chunk)?;
                if !text.is_empty() {
                    let segment = Segment {
                        start: (start as i64 * 100) / vad_rs::SAMPLE_RATE as i64,
                        end: (end as i64 * 100) / vad_rs::SAMPLE_RATE as i64,
                        text,
                    };
                    on_segment(&segment);
                    segments.push(segment);
                }
            }
            on_progress(((end as u64 * 100) / samples.len().max(1) as u64) as i32);
        }
        on_progress(100);
        eprintln!(
            "[sensevoice][long-form] total_ms={} segments={} audio_s={:.3}",
            started.elapsed().as_millis(),
            segments.len(),
            samples.len() as f64 / vad_rs::SAMPLE_RATE as f64
        );
        Ok(Transcription { segments })
    }

    /// One VAD range: fbank, query rows, position encoding, encoder, greedy CTC.
    pub fn transcribe_chunk(&self, samples: &[f32]) -> Result<String> {
        let started = Instant::now();
        let features = fbank::compute(samples)?;
        let width = fbank::FEATURE_SIZE;
        // One prompt row per query token, already in token order.
        let prompt = self.embeddings.len() / width;
        let count = prompt + features.frames;

        // The model was trained with the whole prompt scaled by sqrt(width) before the position
        // encoding is added.
        let scale = (self.info.hidden as f32).sqrt();
        let mut input = Vec::with_capacity(count * width);
        for row in self.embeddings.chunks_exact(width) {
            input.extend(row.iter().map(|value| value * scale));
        }
        input.extend(features.data.iter().map(|value| value * scale));
        encoder::add_posenc(&mut input, count, width);

        unsafe {
            let mut runtime = self.runtime.lock().unwrap_or_else(|poison| poison.into_inner());
            let graph = runtime.build()?;
            let audio = sys::ggml_new_tensor_2d(graph.ctx, sys::ggml_type_GGML_TYPE_F32, width as i64, count as i64);
            let logits = encoder::build(graph.ctx, self, audio, count as i64);
            if (*logits).ne[0] != self.info.vocab || (*logits).ne[1] != count as i64 {
                return Err(Error::InvalidTensor {
                    name: "ctc.ctc_lo.weight".into(),
                    message: format!("logits of [{}, {}] for {count} frames", (*logits).ne[0], (*logits).ne[1]),
                });
            }
            sys::ggml_set_output(logits);
            sys::ggml_build_forward_expand(graph.graph, logits);
            runtime.run(&graph, audio, &input)?;

            let mut scores = vec![0.0f32; self.info.vocab as usize * count];
            sys::ggml_backend_tensor_get(logits, scores.as_mut_ptr().cast(), 0, scores.len() * 4);
            if scores.iter().any(|value| !value.is_finite()) {
                return Err(Error::NonFinite("CTC logits"));
            }
            let text = ctc::decode(
                &scores,
                count,
                self.info.vocab as usize,
                self.info.blank as usize,
                &self.vocab,
            );
            eprintln!(
                "[sensevoice][chunk] frames={} samples={} chars={} elapsed_ms={}",
                features.frames,
                samples.len(),
                text.len(),
                started.elapsed().as_millis()
            );
            Ok(text)
        }
    }
}
