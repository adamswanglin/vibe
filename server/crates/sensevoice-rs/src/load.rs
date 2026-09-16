//! Getting a SenseVoice GGUF off disk and into a [`Model`]: reading the `sv.*` metadata keys, the
//! SentencePiece vocabulary, the query-token embedding table, and the tensor catalog, then copying
//! the weights to a GPU when one is available.
//!
//! Inference itself lives in [`crate::model`]; nothing here runs a graph.

use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr;
use std::sync::Mutex;
use std::time::Instant;

use crate::fbank::FEATURE_SIZE;
use crate::model::{LoadOptions, Model};
use crate::runtime::{init_gpu, Runtime, Weights};
use crate::{sys, Error, Result};

/// GGUF metadata and the derived shapes every graph depends on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    pub architecture: String,
    /// Fbank width, including the LFR stacking: 560 for SenseVoiceSmall.
    pub input_size: i64,
    /// Encoder width, the `sv.output_size` key.
    pub hidden: i64,
    pub heads: i64,
    pub blocks: i64,
    pub tp_blocks: i64,
    /// FSMN kernel width: the number of shifted memory slots summed per layer.
    pub kernel: i64,
    pub vocab: i64,
    pub blank: i64,
    pub tensor_count: usize,
}

impl Model {
    /// Loads with the default options: the first GPU when one exists, the CPU otherwise.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::load_with(path, LoadOptions::default())
    }

    /// Loads the GGUF: metadata and vocabulary first, then one weight buffer on the GPU or the
    /// CPU. The metadata contexts are dropped as soon as the weights have been copied out.
    pub fn load_with(path: impl AsRef<Path>, options: LoadOptions) -> Result<Self> {
        let started = Instant::now();
        let path = path.as_ref().to_string_lossy().into_owned();
        let name = CString::new(path.as_bytes()).map_err(|_| Error::InvalidPath)?;
        let mut meta = ptr::null_mut();
        let gguf = unsafe {
            sys::gguf_init_from_file(
                name.as_ptr(),
                sys::gguf_init_params {
                    no_alloc: true,
                    ctx: &mut meta,
                },
            )
        };
        if gguf.is_null() || meta.is_null() {
            unsafe {
                if !meta.is_null() {
                    sys::ggml_free(meta);
                }
                if !gguf.is_null() {
                    sys::gguf_free(gguf);
                }
            }
            return Err(Error::Load(path));
        }
        let archive = Archive { gguf, meta };
        let info = unsafe { read_info(archive.gguf)? };
        let vocab = unsafe { vocabulary(archive.gguf, info.vocab)? };
        let queries = unsafe { query_tokens(archive.gguf)? };
        let meta_ms = started.elapsed().as_millis();
        let gpu = unsafe { init_gpu(options.no_gpu, options.gpu_device) };
        let backend = if gpu.is_some() { "gpu" } else { "cpu" };
        let runtime = Runtime::new(gpu)?;
        let weights = unsafe { Weights::load(archive.gguf, archive.meta, &path, runtime.buffer_type()) }?;
        let weights_ms = started.elapsed().as_millis() - meta_ms;
        let embeddings = unsafe { embeddings(&weights, &queries, info.input_size)? };
        let model = Self {
            weights,
            info,
            vocab,
            embeddings,
            runtime: Mutex::new(runtime),
        };
        unsafe { model.validate()? };
        eprintln!(
            "[sensevoice][load] elapsed_ms={} tensors={} backend={backend} meta_ms={meta_ms} weights_ms={weights_ms}",
            started.elapsed().as_millis(),
            model.info.tensor_count
        );
        Ok(model)
    }

    /// Reads the metadata without allocating or uploading tensor data.
    pub fn metadata(path: impl AsRef<Path>) -> Result<ModelInfo> {
        let path = CString::new(path.as_ref().to_string_lossy().as_bytes()).map_err(|_| Error::InvalidPath)?;
        let mut weights = ptr::null_mut();
        let gguf = unsafe {
            sys::gguf_init_from_file(
                path.as_ptr(),
                sys::gguf_init_params {
                    no_alloc: true,
                    ctx: &mut weights,
                },
            )
        };
        if gguf.is_null() {
            return Err(Error::Load(path.to_string_lossy().into_owned()));
        }
        let result = unsafe { read_info(gguf) };
        unsafe {
            if !weights.is_null() {
                sys::ggml_free(weights);
            }
            sys::gguf_free(gguf);
        }
        result
    }

    unsafe fn validate(&self) -> Result<()> {
        // The published GGUF calls itself `sensevoice-small` (or `sensevoice`).
        if !self.info.architecture.starts_with("sensevoice") {
            return Err(Error::UnsupportedArchitecture(self.info.architecture.clone()));
        }
        if self.info.input_size != FEATURE_SIZE as i64 {
            return Err(Error::InvalidMetadata {
                key: "sv.input_size",
                message: format!("expected {FEATURE_SIZE}, got {}", self.info.input_size),
            });
        }
        self.validate_tensor_catalog()
    }

    unsafe fn validate_tensor_catalog(&self) -> Result<()> {
        for prefix in self.layer_prefixes() {
            for suffix in [
                "norm1.weight",
                "norm1.bias",
                "self_attn.linear_q_k_v.weight",
                "self_attn.linear_q_k_v.bias",
                "self_attn.linear_out.weight",
                "self_attn.linear_out.bias",
                "self_attn.fsmn_block.weight",
                "norm2.weight",
                "norm2.bias",
                "feed_forward.w_1.weight",
                "feed_forward.w_1.bias",
                "feed_forward.w_2.weight",
                "feed_forward.w_2.bias",
            ] {
                self.require(&format!("{prefix}{suffix}"))?;
            }
        }
        for name in [
            "encoder.after_norm.weight",
            "encoder.after_norm.bias",
            "encoder.tp_norm.weight",
            "encoder.tp_norm.bias",
            "ctc.ctc_lo.weight",
            "ctc.ctc_lo.bias",
            "embed.weight",
        ] {
            self.require(name)?;
        }

        self.expect_shape("encoder.encoders0.0.self_attn.linear_q_k_v.weight", 0, self.info.input_size)?;
        self.expect_shape("encoder.encoders0.0.self_attn.linear_q_k_v.weight", 1, 3 * self.info.hidden)?;
        self.expect_shape("encoder.encoders.0.self_attn.linear_q_k_v.weight", 0, self.info.hidden)?;
        self.expect_shape("encoder.encoders.0.self_attn.linear_q_k_v.weight", 1, 3 * self.info.hidden)?;
        self.expect_shape("encoder.encoders0.0.self_attn.fsmn_block.weight", 0, self.info.hidden)?;
        self.expect_shape("encoder.encoders0.0.self_attn.fsmn_block.weight", 1, self.info.kernel)?;
        self.expect_shape("ctc.ctc_lo.weight", 0, self.info.hidden)?;
        self.expect_shape("ctc.ctc_lo.weight", 1, self.info.vocab)?;
        self.expect_shape("embed.weight", 0, self.info.input_size)?;
        Ok(())
    }

    fn layer_prefixes(&self) -> Vec<String> {
        let mut prefixes = Vec::with_capacity((self.info.blocks + self.info.tp_blocks) as usize);
        prefixes.push("encoder.encoders0.0.".to_owned());
        prefixes.extend((0..self.info.blocks - 1).map(|layer| format!("encoder.encoders.{layer}.")));
        prefixes.extend((0..self.info.tp_blocks).map(|layer| format!("encoder.tp_encoders.{layer}.")));
        prefixes
    }

    unsafe fn require(&self, name: &str) -> Result<()> {
        self.weights
            .tensor(name)
            .map(|_| ())
            .ok_or_else(|| Error::MissingTensor(name.to_owned()))
    }

    unsafe fn expect_shape(&self, name: &str, axis: usize, expected: i64) -> Result<()> {
        let tensor = self
            .weights
            .tensor(name)
            .ok_or_else(|| Error::MissingTensor(name.to_owned()))?;
        let actual = (*tensor).ne[axis];
        if actual != expected {
            return Err(Error::InvalidTensor {
                name: name.to_owned(),
                message: format!("expected {expected} on axis {axis}, got {actual}"),
            });
        }
        Ok(())
    }
}

/// The GGUF reader and the metadata context it filled; both are released once the weights have
/// been copied into their backend buffer.
struct Archive {
    gguf: *mut sys::gguf_context,
    meta: *mut sys::ggml_context,
}

impl Drop for Archive {
    fn drop(&mut self) {
        unsafe {
            sys::gguf_free(self.gguf);
            sys::ggml_free(self.meta);
        }
    }
}

/// Copies the `embed.weight` rows the query tokens name out of the backend buffer. They are prompt
/// tokens, reused unchanged on every chunk, so they do not need to stay on the device.
unsafe fn embeddings(weights: &Weights, queries: &[i64], width: i64) -> Result<Vec<f32>> {
    let tensor = weights
        .tensor("embed.weight")
        .ok_or_else(|| Error::MissingTensor("embed.weight".into()))?;
    if (*tensor).type_ != sys::ggml_type_GGML_TYPE_F32 {
        return Err(Error::InvalidTensor {
            name: "embed.weight".into(),
            message: "expected F32".into(),
        });
    }
    if (*tensor).ne[0] != width {
        return Err(Error::InvalidTensor {
            name: "embed.weight".into(),
            message: format!("expected {width} columns, got {}", (*tensor).ne[0]),
        });
    }
    let rows = (*tensor).ne[1];
    let mut embeddings = Vec::with_capacity(queries.len() * width as usize);
    for &query in queries {
        if query < 0 || query >= rows {
            return Err(Error::InvalidMetadata {
                key: "sv.query_tokens",
                message: format!("token {query} outside an embedding table of {rows}"),
            });
        }
        let offset = query as usize * width as usize * std::mem::size_of::<f32>();
        let mut row = vec![0.0f32; width as usize];
        sys::ggml_backend_tensor_get(tensor, row.as_mut_ptr().cast(), offset, row.len() * 4);
        embeddings.extend_from_slice(&row);
    }
    if embeddings.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFinite("embed.weight"));
    }
    Ok(embeddings)
}

pub(crate) unsafe fn read_info(gguf: *const sys::gguf_context) -> Result<ModelInfo> {
    let input_size = u32_value(gguf, "sv.input_size")? as i64;
    let hidden = u32_value(gguf, "sv.output_size")? as i64;
    let heads = u32_value(gguf, "sv.attention_heads")? as i64;
    let blocks = u32_value(gguf, "sv.num_blocks")? as i64;
    let tp_blocks = u32_value(gguf, "sv.tp_blocks")? as i64;
    let kernel = u32_value(gguf, "sv.kernel_size")? as i64;
    let vocab = u32_value(gguf, "sv.vocab_size")? as i64;
    let blank = u32_value(gguf, "sv.blank_id")? as i64;
    if vocab <= 0 || vocab > 262_144 || blank < 0 || blank >= vocab {
        return Err(Error::InvalidMetadata {
            key: "sv.vocab_size",
            message: format!("vocabulary of {vocab} with blank {blank} is not usable"),
        });
    }
    if hidden <= 0 || heads <= 0 || hidden % heads != 0 || blocks <= 0 || tp_blocks < 0 || kernel <= 0 || kernel % 2 == 0 {
        return Err(Error::InvalidMetadata {
            key: "sv.output_size",
            message: format!("inconsistent encoder configuration: hidden={hidden} heads={heads} blocks={blocks} tp_blocks={tp_blocks} kernel={kernel}"),
        });
    }
    Ok(ModelInfo {
        architecture: string(gguf, "general.architecture")?,
        input_size,
        hidden,
        heads,
        blocks,
        tp_blocks,
        kernel,
        vocab,
        blank,
        tensor_count: sys::gguf_get_n_tensors(gguf) as usize,
    })
}

unsafe fn find_key(gguf: *const sys::gguf_context, name: &'static str) -> Option<i64> {
    let name = CString::new(name).ok()?;
    let id = sys::gguf_find_key(gguf, name.as_ptr());
    (id >= 0).then_some(id)
}

unsafe fn key(gguf: *const sys::gguf_context, name: &'static str) -> Result<i64> {
    find_key(gguf, name).ok_or(Error::MissingMetadata(name))
}

unsafe fn string(gguf: *const sys::gguf_context, name: &'static str) -> Result<String> {
    let value = sys::gguf_get_val_str(gguf, key(gguf, name)?);
    if value.is_null() {
        return Err(Error::InvalidMetadata {
            key: name,
            message: "null string".into(),
        });
    }
    Ok(CStr::from_ptr(value).to_string_lossy().into_owned())
}

unsafe fn u32_value(gguf: *const sys::gguf_context, name: &'static str) -> Result<u32> {
    let id = key(gguf, name)?;
    if sys::gguf_get_kv_type(gguf, id) != sys::gguf_type_GGUF_TYPE_UINT32 {
        return Err(Error::InvalidMetadata {
            key: name,
            message: "expected a uint32".into(),
        });
    }
    Ok(sys::gguf_get_val_u32(gguf, id))
}

unsafe fn vocabulary(gguf: *const sys::gguf_context, size: i64) -> Result<Vec<String>> {
    let id = key(gguf, "sv.vocab")?;
    let count = sys::gguf_get_arr_n(gguf, id);
    if sys::gguf_get_kv_type(gguf, id) != sys::gguf_type_GGUF_TYPE_ARRAY
        || sys::gguf_get_arr_type(gguf, id) != sys::gguf_type_GGUF_TYPE_STRING
        || count != size as usize
    {
        return Err(Error::InvalidMetadata {
            key: "sv.vocab",
            message: format!("{count} string pieces for a vocabulary of {size}"),
        });
    }
    (0..count)
        .map(|index| {
            let piece = sys::gguf_get_arr_str(gguf, id, index);
            if piece.is_null() {
                Err(Error::InvalidMetadata {
                    key: "sv.vocab",
                    message: format!("null piece at {index}"),
                })
            } else {
                Ok(CStr::from_ptr(piece).to_string_lossy().into_owned())
            }
        })
        .collect()
}

unsafe fn query_tokens(gguf: *const sys::gguf_context) -> Result<Vec<i64>> {
    let id = key(gguf, "sv.query_tokens")?;
    let element = sys::gguf_get_arr_type(gguf, id);
    let count = sys::gguf_get_arr_n(gguf, id);
    if sys::gguf_get_kv_type(gguf, id) != sys::gguf_type_GGUF_TYPE_ARRAY
        || !matches!(element, sys::gguf_type_GGUF_TYPE_INT32 | sys::gguf_type_GGUF_TYPE_UINT32)
    {
        return Err(Error::InvalidMetadata {
            key: "sv.query_tokens",
            message: "expected a 32-bit integer array".into(),
        });
    }
    if count == 0 || count > 16 {
        return Err(Error::InvalidMetadata {
            key: "sv.query_tokens",
            message: format!("{count} query tokens"),
        });
    }
    let data = sys::gguf_get_arr_data(gguf, id).cast::<u32>();
    if data.is_null() {
        return Err(Error::InvalidMetadata {
            key: "sv.query_tokens",
            message: "null array".into(),
        });
    }
    // Both int32 and uint32 bit patterns land in the same non-negative id range.
    Ok(std::slice::from_raw_parts(data, count)
        .iter()
        .map(|value| *value as i64)
        .collect())
}
