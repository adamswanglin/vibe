//! SenseVoice ASR on vibe-server's shared GGML runtime.
//!
//! [`Model`] loads a SenseVoice GGUF in-process and runs it on the same backend scheduler as the
//! other native engines, so nothing about it is special-cased outside this crate.

mod ctc;
mod encoder;
mod fbank;
mod load;
mod model;
mod runtime;

pub use ggml_rs_sys as sys;
pub use load::ModelInfo;
pub use model::{LoadOptions, Model, Segment, Transcription};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("model path contains an interior NUL")]
    InvalidPath,
    #[error("failed to load GGUF model: {0}")]
    Load(String),
    #[error("missing GGUF metadata key: {0}")]
    MissingMetadata(&'static str),
    #[error("invalid GGUF metadata {key}: {message}")]
    InvalidMetadata { key: &'static str, message: String },
    #[error("unsupported model architecture {0:?}; expected sensevoice")]
    UnsupportedArchitecture(String),
    #[error("model is missing required tensor {0}")]
    MissingTensor(String),
    #[error("unexpected tensor shape for {name}: {message}")]
    InvalidTensor { name: String, message: String },
    #[error("non-finite values in {0}")]
    NonFinite(&'static str),
    #[error("audio is shorter than one analysis window")]
    AudioTooShort,
    #[error("GGML operation failed: {0}")]
    Ggml(&'static str),
    #[error("VAD failed: {0}")]
    Vad(String),
    #[error("transcription aborted")]
    Aborted,
}

pub type Result<T> = std::result::Result<T, Error>;
