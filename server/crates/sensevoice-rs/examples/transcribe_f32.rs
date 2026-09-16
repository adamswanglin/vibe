//! Smoke test for the engine: transcribe a raw little-endian f32, 16 kHz, mono file.
//!
//! `cargo run -p sensevoice-rs --example transcribe_f32 -- <model.gguf> <audio.f32> [cpu]`

use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let Some(model) = arguments.next() else {
        return Err("usage: transcribe_f32 <model.gguf> <audio.f32> [cpu]".into());
    };
    let Some(audio) = arguments.next() else {
        return Err("usage: transcribe_f32 <model.gguf> <audio.f32> [cpu]".into());
    };
    let no_gpu = arguments.next().is_some();

    let bytes = std::fs::read(audio)?;
    let samples = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();

    let started = Instant::now();
    let model = sensevoice_rs::Model::load_with(PathBuf::from(model), sensevoice_rs::LoadOptions { no_gpu, gpu_device: -1 })?;
    println!("load_ms={}", started.elapsed().as_millis());

    let started = Instant::now();
    let text = model.transcribe_chunk(&samples)?;
    println!("transcribe_ms={} text={text:?}", started.elapsed().as_millis());
    Ok(())
}
