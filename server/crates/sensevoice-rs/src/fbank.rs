//! Kaldi-style 80-bin log-mel frontend with low-frame-rate stacking, bit for bit what FunASR's
//! `funasr-sensevoice` frontend computes: no dither, hamming window, pre-emphasis, samples scaled to
//! the int16 range, and a 7-frame window stepped by 6 frames.

use std::sync::OnceLock;

use crate::{Error, Result};

pub(crate) const FEATURE_SIZE: usize = NMEL * LFR_M;
pub(crate) const WIN_LEN: usize = 400;
const SAMPLE_RATE: f32 = 16_000.0;
const SHIFT: usize = 160;
const FFT_SIZE: usize = 512;
const BINS: usize = FFT_SIZE / 2 + 1;
const NMEL: usize = 80;
const LFR_M: usize = 7;
const LFR_N: usize = 6;
const PRE_EMPHASIS: f32 = 0.97;
const LOW_FREQ: f32 = 20.0;
const HIGH_FREQ: f32 = 8000.0;
const LOG_FLOOR: f32 = 1.1920929e-7;

pub(crate) struct Features {
    /// `frames * FEATURE_SIZE` values, frame-major.
    pub data: Vec<f32>,
    pub frames: usize,
}

struct Frontend {
    window: [f32; WIN_LEN],
    /// `NMEL` triangular filters over the `BINS` non-negative FFT bins.
    filters: Vec<[f32; BINS]>,
}

fn frontend() -> &'static Frontend {
    static FRONTEND: OnceLock<Frontend> = OnceLock::new();
    FRONTEND.get_or_init(|| {
        let mut window = [0.0f32; WIN_LEN];
        for (index, value) in window.iter_mut().enumerate() {
            *value = 0.54 - 0.46 * (2.0 * std::f32::consts::PI * index as f32 / (WIN_LEN - 1) as f32).cos();
        }

        let bin_width = SAMPLE_RATE / FFT_SIZE as f32;
        let low = mel(LOW_FREQ);
        let high = mel(HIGH_FREQ);
        let step = (high - low) / (NMEL as f32 + 1.0);
        let mut filters = vec![[0.0f32; BINS]; NMEL];
        for (index, filter) in filters.iter_mut().enumerate() {
            let left = low + index as f32 * step;
            let center = low + (index as f32 + 1.0) * step;
            let right = low + (index as f32 + 2.0) * step;
            for (bin, weight) in filter.iter_mut().enumerate() {
                let frequency = mel(bin_width * bin as f32);
                if frequency > left && frequency < right {
                    *weight = if frequency <= center {
                        (frequency - left) / (center - left)
                    } else {
                        (right - frequency) / (right - center)
                    };
                }
            }
        }
        Frontend { window, filters }
    })
}

fn mel(frequency: f32) -> f32 {
    1127.0 * (1.0 + frequency / 700.0).ln()
}

/// Extracts `LFR_M`-stacked fbank features, or [`Error::AudioTooShort`] when the chunk cannot fill
/// one analysis window.
pub(crate) fn compute(samples: &[f32]) -> Result<Features> {
    if samples.len() < WIN_LEN {
        return Err(Error::AudioTooShort);
    }
    let frontend = frontend();
    // SenseVoice's frontend works on the int16 amplitude range, not on -1..1.
    let audio: Vec<f32> = samples.iter().map(|sample| sample * 32768.0).collect();

    let total = (audio.len() - WIN_LEN) / SHIFT + 1;
    let mut frames: Vec<[f32; NMEL]> = vec![[0.0; NMEL]; total];
    let mut re = [0.0f32; FFT_SIZE];
    let mut im = [0.0f32; FFT_SIZE];
    for (index, frame) in frames.iter_mut().enumerate() {
        let slice = &audio[index * SHIFT..index * SHIFT + WIN_LEN];
        let mean = slice.iter().map(|sample| *sample as f64).sum::<f64>() / WIN_LEN as f64;
        let mean = mean as f32;

        let mut windowed = [0.0f32; FFT_SIZE];
        for position in 0..WIN_LEN {
            windowed[position] = slice[position] - mean;
        }
        for position in (1..WIN_LEN).rev() {
            windowed[position] -= PRE_EMPHASIS * windowed[position - 1];
        }
        windowed[0] -= PRE_EMPHASIS * windowed[0];
        for position in 0..WIN_LEN {
            re[position] = windowed[position] * frontend.window[position];
        }
        re[WIN_LEN..].fill(0.0);
        im.fill(0.0);
        fft(&mut re, &mut im);

        for (bin, filter) in frontend.filters.iter().enumerate() {
            let mut energy = 0.0f32;
            for (k, weight) in filter.iter().enumerate() {
                if *weight > 0.0 {
                    energy += weight * (re[k] * re[k] + im[k] * im[k]);
                }
            }
            frame[bin] = if energy > LOG_FLOOR { energy } else { LOG_FLOOR }.ln();
        }
    }

    Ok(stack(&frames, total))
}

/// Low-frame-rate stacking: `LFR_M` consecutive frames every `LFR_N` steps, first frame repeated
/// `(LFR_M - 1) / 2` times at the front and the last frame repeated at the end.
fn stack(frames: &[[f32; NMEL]], total: usize) -> Features {
    let centre = (LFR_M - 1) / 2;
    let stacked = (total + LFR_N - 1) / LFR_N;
    let mut sources = Vec::with_capacity(centre + total + LFR_M);
    sources.extend(std::iter::repeat_n(0, centre));
    sources.extend(0..total);
    let needed = stacked.saturating_sub(1) * LFR_N + LFR_M;
    sources.resize(sources.len().max(needed), total.saturating_sub(1));

    let mut data = vec![0.0f32; stacked * FEATURE_SIZE];
    for index in 0..stacked {
        for offset in 0..LFR_M {
            let source = sources[index * LFR_N + offset];
            let destination = index * FEATURE_SIZE + offset * NMEL;
            data[destination..destination + NMEL].copy_from_slice(&frames[source]);
        }
    }
    Features { data, frames: stacked }
}

/// In-place radix-2 FFT over `re`/`im`, `FFT_SIZE` long. The twiddle factors are computed with the
/// same single-precision rounding as the C++ frontend so the features stay comparable.
fn fft(re: &mut [f32; FFT_SIZE], im: &mut [f32; FFT_SIZE]) {
    let n = FFT_SIZE;
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let angle = (-2.0 * std::f64::consts::PI / len as f64) as f32;
        let (sin, cos) = (angle.sin(), angle.cos());
        let half = len / 2;
        let mut start = 0;
        while start < n {
            let (mut cos_k, mut sin_k) = (1.0f32, 0.0f32);
            for k in 0..half {
                let (ur, ui) = (re[start + k], im[start + k]);
                let (vr, vi) = (re[start + k + half], im[start + k + half]);
                let (tr, ti) = (vr * cos_k - vi * sin_k, vr * sin_k + vi * cos_k);
                re[start + k] = ur + tr;
                im[start + k] = ui + ti;
                re[start + k + half] = ur - tr;
                im[start + k + half] = ui - ti;
                let next = cos_k * cos - sin_k * sin;
                sin_k = cos_k * sin + sin_k * cos;
                cos_k = next;
            }
            start += len;
        }
        len <<= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_audio_shorter_than_one_window() {
        assert!(matches!(compute(&[0.1; WIN_LEN - 1]), Err(Error::AudioTooShort)));
    }

    #[test]
    fn stacks_seven_frames_step_six() {
        // 400 + 6 * 160 samples fill exactly two stacked frames.
        let samples = vec![0.1f32; WIN_LEN + 6 * SHIFT];
        let features = compute(&samples).unwrap();
        assert_eq!(features.frames, (7 + 5) / 6);
        assert_eq!(features.data.len(), features.frames * FEATURE_SIZE);
        assert!(features.data.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn loud_frames_stay_finite() {
        // Every frame must clear the zero-padded tail of the FFT input; leaving the previous frame
        // there made loud stretches overflow to infinity.
        let mut samples = vec![0.0f32; WIN_LEN + 40 * SHIFT];
        for (index, sample) in samples.iter_mut().enumerate() {
            if index > WIN_LEN * 2 {
                *sample = if index % 2 == 0 { 0.9 } else { -0.9 };
            }
        }
        let features = compute(&samples).unwrap();
        assert!(features.data.iter().all(|value| value.is_finite()));
        assert!(features.data.iter().all(|value| *value < 40.0));
    }

    #[test]
    fn matches_a_direct_discrete_fourier_transform() {
        let input: Vec<f32> = (0..FFT_SIZE)
            .map(|index| ((index as f32) * 0.11).sin() * 0.5 + (index as f32 * 0.03).cos())
            .collect();
        let mut re = [0.0f32; FFT_SIZE];
        let mut im = [0.0f32; FFT_SIZE];
        re.copy_from_slice(&input);
        fft(&mut re, &mut im);

        for bin in [0usize, 1, 7, 64, 511] {
            let (mut expected_re, mut expected_im) = (0.0f64, 0.0f64);
            for (index, value) in input.iter().enumerate() {
                let angle = -2.0 * std::f64::consts::PI * (bin * index) as f64 / FFT_SIZE as f64;
                expected_re += *value as f64 * angle.cos();
                expected_im += *value as f64 * angle.sin();
            }
            assert!((re[bin] as f64 - expected_re).abs() < 1e-2, "bin {bin} real");
            assert!((im[bin] as f64 - expected_im).abs() < 1e-2, "bin {bin} imag");
        }
    }
}
