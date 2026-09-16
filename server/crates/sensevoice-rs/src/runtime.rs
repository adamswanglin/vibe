//! Device selection, weight buffers, and graph execution on vibe-server's shared GGML runtime.
//!
//! Weights always live in a real backend buffer — the GPU when one is available, the CPU
//! otherwise — and the CPU backend always joins the scheduler, so the few ops a GPU build cannot
//! run fall back instead of failing the whole graph. This is the same shape as the whisper
//! encoder runtime.

use std::ffi::CStr;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::ptr;

use crate::{sys, Error, Result};

/// Graph reservation: the encoder unrolls 70 SAN-M layers at roughly a hundred nodes each.
const MAX_NODES: usize = 16_384;

/// The GPU backend for `gpu_device` (the n-th GPU or IGPU device), or `None` for the CPU. An
/// unpinned device (-1) means the first GPU.
pub(crate) unsafe fn init_gpu(no_gpu: bool, gpu_device: i32) -> Option<sys::ggml_backend_t> {
    if no_gpu {
        return None;
    }
    sys::ggml_backend_load_all();
    let mut index = 0;
    for ordinal in 0..sys::ggml_backend_dev_count() {
        let device = sys::ggml_backend_dev_get(ordinal);
        if !matches!(
            sys::ggml_backend_dev_type(device),
            sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU | sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU
        ) {
            continue;
        }
        if index == gpu_device.max(0) {
            let backend = sys::ggml_backend_dev_init(device, ptr::null());
            if backend.is_null() {
                let name = CStr::from_ptr(sys::ggml_backend_dev_name(device));
                eprintln!("[sensevoice][device] failed to initialize {}", name.to_string_lossy());
                return None;
            }
            return Some(backend);
        }
        index += 1;
    }
    None
}

/// Every weight in one backend buffer, read straight from the GGUF payload — the worker's
/// `load_model_weights`, so the CPU path stays numerically identical to the previous engine.
pub(crate) struct Weights {
    pub(crate) ctx: *mut sys::ggml_context,
    buffer: sys::ggml_backend_buffer_t,
}

impl Weights {
    pub(crate) unsafe fn load(
        gguf: *mut sys::gguf_context,
        meta: *mut sys::ggml_context,
        path: &str,
        buffer_type: sys::ggml_backend_buffer_type_t,
    ) -> Result<Self> {
        let count = sys::gguf_get_n_tensors(gguf);
        let ctx = sys::ggml_init(sys::ggml_init_params {
            mem_size: sys::ggml_tensor_overhead() * (count as usize + 32),
            mem_buffer: ptr::null_mut(),
            no_alloc: true,
        });
        if ctx.is_null() {
            return Err(Error::Ggml("weight context"));
        }
        for index in 0..count {
            let name = sys::gguf_get_tensor_name(gguf, index);
            let layout = sys::ggml_get_tensor(meta, name);
            if layout.is_null() {
                sys::ggml_free(ctx);
                return Err(Error::Load(format!("missing tensor metadata at {index}")));
            }
            let tensor = sys::ggml_dup_tensor(ctx, layout);
            sys::ggml_set_name(tensor, name);
        }
        let buffer = sys::ggml_backend_alloc_ctx_tensors_from_buft(ctx, buffer_type);
        if buffer.is_null() {
            sys::ggml_free(ctx);
            return Err(Error::Ggml("weight buffer"));
        }
        sys::ggml_backend_buffer_set_usage(buffer, sys::ggml_backend_buffer_usage_GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        let weights = Self { ctx, buffer };
        match weights.fill(gguf, path) {
            Ok(()) => Ok(weights),
            Err(error) => {
                drop(weights);
                Err(error)
            }
        }
    }

    unsafe fn fill(&self, gguf: *mut sys::gguf_context, path: &str) -> Result<()> {
        let mut file = BufReader::new(File::open(path).map_err(|error| Error::Load(format!("{path}: {error}")))?);
        let base = sys::gguf_get_data_offset(gguf) as u64;
        let mut bytes = Vec::new();
        for index in 0..sys::gguf_get_n_tensors(gguf) {
            let name = sys::gguf_get_tensor_name(gguf, index);
            let tensor = sys::ggml_get_tensor(self.ctx, name);
            if tensor.is_null() {
                return Err(Error::Load(format!("missing tensor layout at {index}")));
            }
            let size = sys::ggml_nbytes(tensor);
            bytes.resize(size, 0);
            let offset = base + sys::gguf_get_tensor_offset(gguf, index) as u64;
            file.seek(SeekFrom::Start(offset))
                .and_then(|_| file.read_exact(&mut bytes))
                .map_err(|error| Error::Load(format!("{path}: {error}")))?;
            sys::ggml_backend_tensor_set(tensor, bytes.as_ptr().cast(), 0, size);
        }
        Ok(())
    }

    pub(crate) unsafe fn tensor(&self, name: &str) -> Option<*mut sys::ggml_tensor> {
        let name = std::ffi::CString::new(name).ok()?;
        let tensor = sys::ggml_get_tensor(self.ctx, name.as_ptr());
        (!tensor.is_null()).then_some(tensor)
    }
}

impl Drop for Weights {
    fn drop(&mut self) {
        unsafe {
            sys::ggml_backend_buffer_free(self.buffer);
            sys::ggml_free(self.ctx);
        }
    }
}

unsafe impl Send for Weights {}

/// The backend set, one scheduler, and one scratch arena. The graph depends on the frame count, so
/// it is rebuilt per audio chunk over the same arena.
pub(crate) struct Runtime {
    backends: Vec<sys::ggml_backend_t>,
    scheduler: sys::ggml_backend_sched_t,
    scratch: Vec<u8>,
}

unsafe impl Send for Runtime {}

impl Runtime {
    pub(crate) fn new(gpu: Option<sys::ggml_backend_t>) -> Result<Self> {
        unsafe {
            sys::ggml_backend_load_all();
            let cpu = sys::ggml_backend_init_by_type(sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_CPU, ptr::null());
            if cpu.is_null() {
                return Err(Error::Ggml("cpu backend"));
            }
            let mut backends = gpu.into_iter().collect::<Vec<_>>();
            backends.push(cpu);
            // `op_offload` lets the scheduler move the ops a backend cannot run to the next one.
            let scheduler = sys::ggml_backend_sched_new(
                backends.as_mut_ptr(),
                ptr::null_mut(),
                backends.len() as i32,
                MAX_NODES,
                false,
                true,
            );
            if scheduler.is_null() {
                for backend in backends {
                    sys::ggml_backend_free(backend);
                }
                return Err(Error::Ggml("backend scheduler"));
            }
            set_threads(&backends, threads());
            let scratch = vec![
                0u8;
                sys::ggml_tensor_overhead() * MAX_NODES
                    + sys::ggml_graph_overhead_custom(MAX_NODES, false)
            ];
            Ok(Self {
                backends,
                scheduler,
                scratch,
            })
        }
    }

    /// The backend the weights go on: the GPU when there is one, the CPU otherwise.
    pub(crate) fn primary(&self) -> sys::ggml_backend_t {
        self.backends[0]
    }

    pub(crate) fn buffer_type(&self) -> sys::ggml_backend_buffer_type_t {
        unsafe { sys::ggml_backend_get_default_buffer_type(self.primary()) }
    }

    /// A fresh graph over the reusable scratch arena.
    pub(crate) unsafe fn build(&mut self) -> Result<Graph> {
        sys::ggml_backend_sched_reset(self.scheduler);
        let ctx = sys::ggml_init(sys::ggml_init_params {
            mem_size: self.scratch.len(),
            mem_buffer: self.scratch.as_mut_ptr().cast(),
            no_alloc: true,
        });
        if ctx.is_null() {
            return Err(Error::Ggml("graph context"));
        }
        let graph = sys::ggml_new_graph_custom(ctx, MAX_NODES, false);
        if graph.is_null() {
            sys::ggml_free(ctx);
            return Err(Error::Ggml("graph"));
        }
        Ok(Graph { ctx, graph })
    }

    /// Allocates `graph`, uploads `input`, and runs it to completion.
    pub(crate) unsafe fn run(&mut self, graph: &Graph, input: *mut sys::ggml_tensor, values: &[f32]) -> Result<()> {
        sys::ggml_set_input(input);
        if !sys::ggml_backend_sched_alloc_graph(self.scheduler, graph.graph) {
            return Err(Error::Ggml("ggml_backend_sched_alloc_graph"));
        }
        sys::ggml_backend_tensor_set(input, values.as_ptr().cast(), 0, std::mem::size_of_val(values));
        let status = sys::ggml_backend_sched_graph_compute(self.scheduler, graph.graph);
        sys::ggml_backend_sched_reset(self.scheduler);
        if status == sys::ggml_status_GGML_STATUS_SUCCESS {
            Ok(())
        } else {
            Err(Error::Ggml("ggml_backend_sched_graph_compute"))
        }
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        unsafe {
            sys::ggml_backend_sched_free(self.scheduler);
            for backend in &self.backends {
                sys::ggml_backend_free(*backend);
            }
        }
    }
}

pub(crate) struct Graph {
    pub(crate) ctx: *mut sys::ggml_context,
    pub(crate) graph: *mut sys::ggml_cgraph,
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe { sys::ggml_free(self.ctx) };
    }
}

fn threads() -> i32 {
    std::thread::available_parallelism()
        .map(|count| (count.get() * 4 / 5).max(1))
        .unwrap_or(4) as i32
}

/// Sets the thread count through the backend registry: the direct
/// `ggml_backend_cpu_set_n_threads` symbol does not exist in `GGML_BACKEND_DL` builds, where the
/// CPU backend is a loadable module.
unsafe fn set_threads(backends: &[sys::ggml_backend_t], n_threads: i32) {
    for &backend in backends {
        let device = sys::ggml_backend_get_device(backend);
        if device.is_null() {
            continue;
        }
        let registry = sys::ggml_backend_dev_backend_reg(device);
        if registry.is_null() {
            continue;
        }
        let address = sys::ggml_backend_reg_get_proc_address(registry, c"ggml_backend_set_n_threads".as_ptr());
        if address.is_null() {
            continue;
        }
        let set_n_threads: sys::ggml_backend_set_n_threads_t = std::mem::transmute(address);
        if let Some(set_n_threads) = set_n_threads {
            set_n_threads(backend, n_threads);
        }
    }
}
