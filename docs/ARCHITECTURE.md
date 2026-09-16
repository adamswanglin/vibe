# Vibe Architecture

## Overview

Vibe is a desktop transcription app built with **Tauri** (Rust + TypeScript frontend).

## Components

### Desktop App (`desktop/`)

- **Frontend**: TypeScript + React (UI)
- **Backend**: Rust/Tauri (`desktop/src-tauri/`)
- Handles UI, file management, settings, analytics
- Spawns and communicates with the server via local HTTP

### Server (`server/`)

- **Language**: Rust on ggml (`whisper-rs`, `parakeet-rs`, `nemotron-rs`, `sensevoice-rs`, `vad-rs`, `diarize-rs`)
- **Location**: `server/` in this repository, its own Cargo workspace; crate and binary `vibe-server`
- **Purpose**: single local process for audio transcription, model loading, streaming, and diarization, behind an OpenAI-compatible HTTP API
- Bundled as the `vibe-server` sidecar with the desktop app
- Diarization is in-process via `diarize-rs`; Vibe does not bundle or spawn a separate binary
- SenseVoice native ASR is in-process through `sensevoice-rs`: a kaldi fbank, the SANM encoder, and greedy CTC decoding on ggml, with no sidecar and no separate protocol
- SenseVoice follows the same device rule as whisper: the "Transcribe on the CPU" setting pins it to the CPU, an unpinned device means the default GPU, and the app's `gpu_device` picks the GPU. Ops the GPU backend lacks — SANM's `pad_ext`, for one, on Metal — fall back to the CPU through the ggml backend scheduler
- SenseVoice models are plain GGUFs: the server reads `general.architecture` to pick the engine, the same way it recognises Parakeet and Nemotron
- **Build**: `server-libs.yml` publishes the prebuilt ggml libraries (`libraries-ggml-<version>-r<revision>`, inputs under `server/libs/`); `server-release.yml` publishes `server-v*` prereleases; `server-test.yml` runs a release across platforms
- **Distribution**: the app downloads the release named by `.server-version` at build time (`setup` task in the root `chorefile`); `chore server-build` stages an in-tree build instead

### FFmpeg Helper

- macOS and Windows builds also bundle `ffmpeg` from the server release archives
- Vibe passes its path to the server with `VIBE_SERVER_FFMPEG_PATH`

### Build Flow

1. Vibe CI runs `chore setup <target-triple>`
2. It downloads the prebuilt `vibe-server` from the `server-v*` release in `.server-version`
3. Binaries placed in `desktop/src-tauri/binaries/`
4. Tauri bundles `vibe-server` and, where configured, `ffmpeg` into the final app

## Key Point for Agents

Native runtime compatibility issues for transcription usually come from the server or its linked ggml libraries, not the Vibe UI code.

To fix one, change `server/` (a ggml fix goes in `server/libs/patches/` with a revision bump), tag a `server-v*` release, then bump `.server-version` here.
