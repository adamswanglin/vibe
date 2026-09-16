//! The SAN-M encoder graph: four query rows followed by the fbank frames go in, CTC logits come
//! out. Every op, view, and scale factor mirrors FunASR's `funasr-sensevoice` reference
//! implementation, so the logits stay directly comparable with it.

use crate::model::Model;
use crate::sys;

const LN_EPS: f32 = 1e-5;

pub(crate) type Context = *mut sys::ggml_context;
pub(crate) type Tensor = *mut sys::ggml_tensor;

/// Adds the sinusoidal position encoding, one wave per position starting at one: the odd half of
/// the row is `sin`, the even half `cos`, with the frequency step `ln(10000) / (width / 2 - 1)`.
pub(crate) fn add_posenc(data: &mut [f32], frames: usize, width: usize) {
    let step = 10_000f64.ln() / (width as f64 / 2.0 - 1.0);
    for frame in 0..frames {
        let position = (frame + 1) as f64;
        for index in 0..width / 2 {
            let scaled = position * (-(index as f64) * step).exp();
            data[frame * width + index] += scaled.sin() as f32;
            data[frame * width + width / 2 + index] += scaled.cos() as f32;
        }
    }
}

/// Builds the whole encoder and the CTC projection on top of it.
pub(crate) unsafe fn build(ctx: Context, model: &Model, input: Tensor, frames: i64) -> Tensor {
    let info = model.info();
    // `encoders0` is the input projection layer: it normalizes over the fbank width and skips the
    // residual because its output is what the following layers add to.
    let mut hidden = layer(ctx, model, "encoder.encoders0.0.", input, frames, false);
    for index in 0..info.blocks - 1 {
        hidden = layer(ctx, model, &format!("encoder.encoders.{index}."), hidden, frames, true);
    }
    hidden = lnorm(
        ctx,
        hidden,
        weight(model, "encoder.after_norm.weight"),
        weight(model, "encoder.after_norm.bias"),
    );
    for index in 0..info.tp_blocks {
        hidden = layer(ctx, model, &format!("encoder.tp_encoders.{index}."), hidden, frames, true);
    }
    hidden = lnorm(ctx, hidden, weight(model, "encoder.tp_norm.weight"), weight(model, "encoder.tp_norm.bias"));
    linear(
        ctx,
        weight(model, "ctc.ctc_lo.weight"),
        weight(model, "ctc.ctc_lo.bias"),
        hidden,
    )
}

unsafe fn weight(model: &Model, name: &str) -> Tensor {
    model.tensor(name).expect("validated tensor catalog")
}

unsafe fn linear(ctx: Context, weight: Tensor, bias: Tensor, x: Tensor) -> Tensor {
    sys::ggml_add(ctx, sys::ggml_mul_mat(ctx, weight, x), bias)
}

unsafe fn lnorm(ctx: Context, x: Tensor, gain: Tensor, bias: Tensor) -> Tensor {
    sys::ggml_add(ctx, sys::ggml_mul(ctx, sys::ggml_norm(ctx, x, LN_EPS), gain), bias)
}

unsafe fn layer(ctx: Context, model: &Model, prefix: &str, x: Tensor, frames: i64, residual: bool) -> Tensor {
    let normed = lnorm(
        ctx,
        x,
        weight(model, &format!("{prefix}norm1.weight")),
        weight(model, &format!("{prefix}norm1.bias")),
    );
    let attended = attention(ctx, model, &format!("{prefix}self_attn."), normed, frames);
    let attended = if residual { sys::ggml_add(ctx, x, attended) } else { attended };
    let normed = lnorm(
        ctx,
        attended,
        weight(model, &format!("{prefix}norm2.weight")),
        weight(model, &format!("{prefix}norm2.bias")),
    );
    let feed = linear(
        ctx,
        weight(model, &format!("{prefix}feed_forward.w_1.weight")),
        weight(model, &format!("{prefix}feed_forward.w_1.bias")),
        normed,
    );
    let feed = sys::ggml_relu(ctx, feed);
    let feed = linear(
        ctx,
        weight(model, &format!("{prefix}feed_forward.w_2.weight")),
        weight(model, &format!("{prefix}feed_forward.w_2.bias")),
        feed,
    );
    sys::ggml_add(ctx, attended, feed)
}

/// Multi-head self attention over the full chunk, with the FSMN memory added to the projection.
unsafe fn attention(ctx: Context, model: &Model, prefix: &str, x: Tensor, frames: i64) -> Tensor {
    let info = model.info();
    let width = info.hidden;
    let per_head = width / info.heads;
    let kernel = info.kernel;

    let qkv = linear(
        ctx,
        weight(model, &format!("{prefix}linear_q_k_v.weight")),
        weight(model, &format!("{prefix}linear_q_k_v.bias")),
        x,
    );
    let stride = (*qkv).nb[1];
    let offset = width as usize * std::mem::size_of::<f32>();
    let q = sys::ggml_cont(ctx, sys::ggml_view_2d(ctx, qkv, width, frames, stride, 0));
    let k = sys::ggml_cont(ctx, sys::ggml_view_2d(ctx, qkv, width, frames, stride, offset));
    let v = sys::ggml_cont(ctx, sys::ggml_view_2d(ctx, qkv, width, frames, stride, 2 * offset));

    // FSMN memory: `kernel` shifted copies of v, zero padded by (kernel - 1) / 2 frames on each
    // side, each weighted by one column of `fsmn_block`.
    let pad = (kernel - 1) as i32 / 2;
    let padded = sys::ggml_pad_ext(ctx, v, 0, 0, pad, pad, 0, 0, 0, 0);
    let block = weight(model, &format!("{prefix}fsmn_block.weight"));
    let mut memory = v;
    for column in 0..kernel as usize {
        let stride = (*padded).nb[1];
        let slice = sys::ggml_view_2d(ctx, padded, width, frames, stride, column * stride);
        let gain = sys::ggml_view_1d(ctx, block, width, column * (*block).nb[1]);
        memory = sys::ggml_add(ctx, memory, sys::ggml_mul(ctx, sys::ggml_cont(ctx, slice), gain));
    }

    let q = sys::ggml_permute(ctx, sys::ggml_reshape_3d(ctx, q, per_head, info.heads, frames), 0, 2, 1, 3);
    let k = sys::ggml_permute(ctx, sys::ggml_reshape_3d(ctx, k, per_head, info.heads, frames), 0, 2, 1, 3);
    let vh = sys::ggml_cont(
        ctx,
        sys::ggml_permute(ctx, sys::ggml_reshape_3d(ctx, v, per_head, info.heads, frames), 1, 2, 0, 3),
    );
    let scores = sys::ggml_soft_max(
        ctx,
        sys::ggml_scale(
            ctx,
            sys::ggml_mul_mat(ctx, k, q),
            1.0 / (per_head as f32).sqrt(),
        ),
    );
    let mixed = sys::ggml_cont_2d(
        ctx,
        sys::ggml_permute(ctx, sys::ggml_mul_mat(ctx, vh, scores), 0, 2, 1, 3),
        width,
        frames,
    );
    let out = linear(
        ctx,
        weight(model, &format!("{prefix}linear_out.weight")),
        weight(model, &format!("{prefix}linear_out.bias")),
        mixed,
    );
    sys::ggml_add(ctx, out, memory)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_encoding_starts_at_position_one() {
        let width = 8;
        let mut data = vec![0.0f32; 2 * width];
        add_posenc(&mut data, 2, width);
        let step = 10_000f64.ln() / (width as f64 / 2.0 - 1.0);
        for frame in 0..2 {
            for index in 0..width / 2 {
                let angle = (frame + 1) as f64 * (-(index as f64) * step).exp();
                assert!((data[frame * width + index] as f64 - angle.sin()).abs() < 1e-6);
                assert!((data[frame * width + width / 2 + index] as f64 - angle.cos()).abs() < 1e-6);
            }
        }
    }
}
