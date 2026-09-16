use anyhow::{ensure, Context as _};
use whisper_rs::{Segment, StreamCallbacks, TranscribeOptions, TranscribeResult};

use super::EngineCapabilities;

pub struct SenseVoice {
    model: sensevoice_rs::Model,
    vad: Option<(String, vad_rs::Vad)>,
}

/// The requested device is outside the range the SANM encoder accepts; the caller reports it as a
/// bad request rather than a server fault.
#[derive(Debug, thiserror::Error)]
#[error("gpu_device must be -1 (CPU) or a GPU index 0..=15")]
pub struct InvalidDevice;

impl SenseVoice {
    pub fn load(path: &str, no_gpu: bool, gpu_device: i32) -> anyhow::Result<Self> {
        let model = sensevoice_rs::Model::load_with(path, sensevoice_rs::LoadOptions { no_gpu, gpu_device })
            .context("failed to load SenseVoice model")?;
        Ok(Self { model, vad: None })
    }

    pub fn capabilities(&self) -> EngineCapabilities {
        capabilities()
    }

    pub fn transcribe(
        &mut self,
        samples: &[f32],
        options: TranscribeOptions,
        callbacks: StreamCallbacks<'_>,
    ) -> anyhow::Result<TranscribeResult> {
        validate_options(&options)?;
        let StreamCallbacks {
            mut on_progress,
            mut on_segment,
            mut should_abort,
        } = callbacks;
        // Split the borrows: the model walks the ranges the cached VAD reports.
        let Self { model, vad } = self;
        let vad_model_path = options
            .vad_model_path
            .as_deref()
            .context("vad_model_path is required for SenseVoice")?;
        if vad.as_ref().is_none_or(|(cached, _)| cached != vad_model_path) {
            *vad = Some((
                vad_model_path.to_owned(),
                vad_rs::Vad::new(vad_model_path, vad_options())?,
            ));
        }
        let result = model
            .transcribe_with(
                &mut vad.as_mut().expect("VAD initialized").1,
                samples,
                || should_abort.as_mut().is_some_and(|callback| callback()),
                |transcription| {
                    if let Some(callback) = on_segment.as_mut() {
                        callback(segment(transcription));
                    }
                },
                |progress| {
                    if let Some(callback) = on_progress.as_mut() {
                        callback(progress);
                    }
                },
            )
            .context("SenseVoice inference failed")?;
        Ok(TranscribeResult {
            segments: result.segments.iter().map(segment).collect(),
        })
    }
}

fn segment(transcription: &sensevoice_rs::Segment) -> Segment {
    Segment {
        start: transcription.start,
        end: transcription.end,
        text: transcription.text.clone(),
        no_speech_prob: 0.0,
    }
}

fn vad_options() -> vad_rs::Options {
    vad_rs::Options {
        max_chunk_ms: 30_000,
        overlap_ms: 0,
        ..vad_rs::Options::default()
    }
}

pub fn capabilities() -> EngineCapabilities {
    EngineCapabilities {
        engine: "sensevoice".to_owned(),
        requires_vad: true,
        languages: Vec::new(),
        language_detection: true,
        streaming: false,
        translation: false,
        timestamps: true,
        text_prompts: false,
    }
}

pub fn validate_device(gpu_device: i32) -> Result<(), InvalidDevice> {
    // The index addresses the GPU the encoder runs on; a build without a GPU backend falls back to
    // the CPU with a stderr note instead of failing.
    if (-1..=15).contains(&gpu_device) {
        Ok(())
    } else {
        Err(InvalidDevice)
    }
}

pub fn validate_options(options: &TranscribeOptions) -> anyhow::Result<()> {
    ensure!(
        matches!(options.language.as_deref(), None | Some("" | "auto")),
        "SenseVoice only supports automatic language detection; language must be 'auto' or omitted"
    );
    ensure!(!options.translate, "SenseVoice does not support translate");
    ensure!(options.prompt.is_none(), "SenseVoice does not support prompt");
    ensure!(!options.word_timestamps, "SenseVoice does not support word_timestamps");
    ensure!(options.max_segment_len <= 0, "SenseVoice does not support max_segment_len > 0");
    ensure!(options.temperature == 0.0, "SenseVoice requires temperature = 0");
    ensure!(options.max_text_ctx <= 0, "SenseVoice does not support max_text_ctx > 0");
    ensure!(options.sampling_greedy, "SenseVoice does not support beam search");
    ensure!(
        matches!(options.threads, 0 | 8),
        "SenseVoice uses 8 CPU threads; n_threads must be 0 or 8"
    );
    for (name, value) in [("best_of", options.best_of), ("beam_size", options.beam_size)] {
        ensure!(
            matches!(value, 0 | 5),
            "SenseVoice does not support custom {name}; omit it or use the default (0 or 5)"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capabilities_and_vad_are_conservative() {
        let caps = capabilities();
        assert_eq!(caps.engine, "sensevoice");
        assert!(caps.requires_vad && caps.language_detection && caps.timestamps);
        assert!(caps.languages.is_empty());
        assert!(!caps.streaming && !caps.translation && !caps.text_prompts);
        assert_eq!(vad_options().max_chunk_ms, 30_000);
        assert_eq!(vad_options().overlap_ms, 0);
        assert!(validate_device(-1).is_ok());
        assert!(validate_device(0).is_ok());
        assert!(validate_device(-2).is_err());
        assert!(validate_device(16).is_err());
    }

    #[test]
    fn rejects_unsupported_options_even_with_detect_language() {
        let defaults = TranscribeOptions::default();
        assert!(validate_options(&defaults).is_ok());
        for options in [
            TranscribeOptions {
                language: Some("en".into()),
                detect_language: true,
                ..defaults.clone()
            },
            TranscribeOptions {
                translate: true,
                ..defaults.clone()
            },
            TranscribeOptions {
                prompt: Some("hello".into()),
                ..defaults.clone()
            },
            TranscribeOptions {
                word_timestamps: true,
                ..defaults.clone()
            },
            TranscribeOptions {
                max_segment_len: 1,
                ..defaults.clone()
            },
            TranscribeOptions {
                temperature: 0.1,
                ..defaults.clone()
            },
            TranscribeOptions {
                temperature: f32::NAN,
                ..defaults.clone()
            },
            TranscribeOptions {
                max_text_ctx: 1,
                ..defaults.clone()
            },
            TranscribeOptions {
                sampling_greedy: false,
                ..defaults.clone()
            },
            TranscribeOptions {
                threads: 4,
                ..defaults.clone()
            },
            TranscribeOptions {
                best_of: 1,
                ..defaults.clone()
            },
            TranscribeOptions {
                beam_size: 1,
                ..defaults.clone()
            },
        ] {
            assert!(validate_options(&options).is_err(), "{options:?}");
        }
        assert!(validate_options(&TranscribeOptions {
            language: Some("auto".into()),
            threads: 8,
            best_of: 5,
            beam_size: 5,
            ..defaults
        })
        .is_ok());
    }
}
