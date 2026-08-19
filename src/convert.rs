//! YUYV 4:2:2 <-> RGB conversion.
//!
//! # Why the matrix is selectable
//!
//! There is no reliable way to tell a V4L2 consumer which YCbCr matrix a stream
//! uses. The device here *declares* BT.601 limited range, and OBS decodes it as
//! BT.709 full range regardless.
//!
//! Camera pixels survive that mismatch: they are decoded and re-encoded with the
//! same matrix, so they reach the consumer byte-identical to the original feed.
//! Only colours this program *introduces* — the greenscreen key and background
//! images — land wrong. Encoding `#00FF00` as BT.601 limited and decoding it as
//! BT.709 full yields exactly `#00CB08`, which is what OBS reports.
//!
//! So the matrix must match whatever the consumer assumes.

use crate::cli::Colorimetry;

/// Luma coefficients and quantisation range for one YCbCr convention.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix {
    kr: f32,
    kb: f32,
    full_range: bool,
}

impl Matrix {
    pub const BT601_LIMITED: Matrix = Matrix { kr: 0.299, kb: 0.114, full_range: false };
    pub const BT601_FULL: Matrix = Matrix { kr: 0.299, kb: 0.114, full_range: true };
    pub const BT709_LIMITED: Matrix = Matrix { kr: 0.2126, kb: 0.0722, full_range: false };
    pub const BT709_FULL: Matrix = Matrix { kr: 0.2126, kb: 0.0722, full_range: true };

    pub fn from_cli(c: Colorimetry) -> Matrix {
        match c {
            Colorimetry::Bt601Limited => Matrix::BT601_LIMITED,
            Colorimetry::Bt601Full => Matrix::BT601_FULL,
            Colorimetry::Bt709Limited => Matrix::BT709_LIMITED,
            Colorimetry::Bt709Full => Matrix::BT709_FULL,
        }
    }

    #[inline]
    fn kg(&self) -> f32 {
        1.0 - self.kr - self.kb
    }

    #[inline]
    pub fn yuv_to_rgb(&self, y: u8, u: u8, v: u8) -> [u8; 3] {
        let (y, cscale) = if self.full_range {
            (y as f32, 1.0)
        } else {
            ((y as f32 - 16.0) * 255.0 / 219.0, 255.0 / 224.0)
        };
        let u = (u as f32 - 128.0) * cscale;
        let v = (v as f32 - 128.0) * cscale;
        let r = y + 2.0 * (1.0 - self.kr) * v;
        let b = y + 2.0 * (1.0 - self.kb) * u;
        let g = (y - self.kr * r - self.kb * b) / self.kg();
        [clamp_u8(r), clamp_u8(g), clamp_u8(b)]
    }

    #[inline]
    pub fn rgb_to_yuv(&self, r: u8, g: u8, b: u8) -> [u8; 3] {
        let (r, g, b) = (r as f32, g as f32, b as f32);
        let y = self.kr * r + self.kg() * g + self.kb * b;
        let u = (b - y) / (2.0 * (1.0 - self.kb));
        let v = (r - y) / (2.0 * (1.0 - self.kr));
        if self.full_range {
            [clamp_u8(y), clamp_u8(u + 128.0), clamp_u8(v + 128.0)]
        } else {
            [
                clamp_u8(y * 219.0 / 255.0 + 16.0),
                clamp_u8(u * 224.0 / 255.0 + 128.0),
                clamp_u8(v * 224.0 / 255.0 + 128.0),
            ]
        }
    }
}

/// Convenience wrappers using the historical default, BT.601 limited.
#[inline]
pub fn yuv_to_rgb(y: u8, u: u8, v: u8) -> [u8; 3] {
    Matrix::BT601_LIMITED.yuv_to_rgb(y, u, v)
}

#[inline]
pub fn rgb_to_yuv(r: u8, g: u8, b: u8) -> [u8; 3] {
    Matrix::BT601_LIMITED.rgb_to_yuv(r, g, b)
}

#[inline]
fn clamp_u8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

/// YUYV 4:2:2 -> planar RGB, normalized to [0,1], laid out NCHW as the model
/// expects. `out` must be `3 * width * height` long.
pub fn yuyv_to_rgb_f32_nchw(
    yuyv: &[u8],
    width: usize,
    height: usize,
    m: Matrix,
    out: &mut [f32],
) {
    let plane = width * height;
    debug_assert_eq!(out.len(), plane * 3);
    debug_assert_eq!(yuyv.len(), plane * 2);

    for row in 0..height {
        let src = row * width * 2;
        let dst = row * width;
        for pair in 0..width / 2 {
            let i = src + pair * 4;
            let (y0, u, y1, v) = (yuyv[i], yuyv[i + 1], yuyv[i + 2], yuyv[i + 3]);
            let p0 = m.yuv_to_rgb(y0, u, v);
            let p1 = m.yuv_to_rgb(y1, u, v);
            let o = dst + pair * 2;
            for c in 0..3 {
                out[c * plane + o] = p0[c] as f32 / 255.0;
                out[c * plane + o + 1] = p1[c] as f32 / 255.0;
            }
        }
    }
}

/// Interleaved RGB8 -> YUYV 4:2:2. Chroma is averaged across each pixel pair.
/// `out` must be `2 * width * height` long.
pub fn rgb_to_yuyv(rgb: &[u8], width: usize, height: usize, m: Matrix, out: &mut [u8]) {
    debug_assert_eq!(rgb.len(), width * height * 3);
    debug_assert_eq!(out.len(), width * height * 2);

    for row in 0..height {
        let src = row * width * 3;
        let dst = row * width * 2;
        for pair in 0..width / 2 {
            let a = src + pair * 6;
            let p0 = m.rgb_to_yuv(rgb[a], rgb[a + 1], rgb[a + 2]);
            let p1 = m.rgb_to_yuv(rgb[a + 3], rgb[a + 4], rgb[a + 5]);
            let o = dst + pair * 4;
            out[o] = p0[0];
            out[o + 1] = ((p0[1] as u16 + p1[1] as u16) / 2) as u8;
            out[o + 2] = p1[0];
            out[o + 3] = ((p0[2] as u16 + p1[2] as u16) / 2) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact mismatch observed against OBS: our BT.601 limited encoding of
    /// pure green, decoded as BT.709 full range, is #00CB08. This pins the
    /// behaviour that made `--colorimetry` necessary.
    #[test]
    fn documented_obs_mismatch_reproduces() {
        let yuv = Matrix::BT601_LIMITED.rgb_to_yuv(0, 255, 0);
        let seen = Matrix::BT709_FULL.yuv_to_rgb(yuv[0], yuv[1], yuv[2]);
        assert_eq!(seen, [0x00, 0xcb, 0x08], "expected the reported #00cb08");
    }

    /// Matching matrices on both sides is what actually fixes it.
    #[test]
    fn matched_matrix_round_trips_key_colours() {
        for m in [
            Matrix::BT601_LIMITED,
            Matrix::BT601_FULL,
            Matrix::BT709_LIMITED,
            Matrix::BT709_FULL,
        ] {
            for colour in [[0u8, 255, 0], [255, 0, 0], [0, 0, 255], [255, 255, 255]] {
                let yuv = m.rgb_to_yuv(colour[0], colour[1], colour[2]);
                let back = m.yuv_to_rgb(yuv[0], yuv[1], yuv[2]);
                for i in 0..3 {
                    assert!(
                        (back[i] as i32 - colour[i] as i32).abs() <= 3,
                        "{m:?}: {colour:?} -> {back:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn grey_round_trips() {
        // Mid grey: equal RGB should map to neutral chroma and back.
        let [y, u, v] = rgb_to_yuv(128, 128, 128);
        let [r, g, b] = yuv_to_rgb(y, u, v);
        assert!((r as i32 - 128).abs() <= 2, "r={r}");
        assert!((g as i32 - 128).abs() <= 2, "g={g}");
        assert!((b as i32 - 128).abs() <= 2, "b={b}");
    }

    #[test]
    fn primaries_round_trip_within_tolerance() {
        for colour in [
            [255u8, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [0, 0, 0],
            [255, 255, 255],
        ] {
            let [y, u, v] = rgb_to_yuv(colour[0], colour[1], colour[2]);
            let got = yuv_to_rgb(y, u, v);
            for i in 0..3 {
                assert!(
                    (got[i] as i32 - colour[i] as i32).abs() <= 4,
                    "{colour:?} -> {got:?} channel {i}"
                );
            }
        }
    }

    #[test]
    fn yuyv_to_nchw_is_planar_and_normalized() {
        // 2x1 image, one YUYV macropixel of white.
        let [y, u, v] = rgb_to_yuv(255, 255, 255);
        let yuyv = [y, u, y, v];
        let mut out = vec![0f32; 3 * 2];
        yuyv_to_rgb_f32_nchw(&yuyv, 2, 1, Matrix::BT601_LIMITED, &mut out);
        for value in &out {
            assert!(*value > 0.95, "expected near 1.0, got {value}");
            assert!(*value <= 1.0, "must be normalized, got {value}");
        }
    }

    #[test]
    fn rgb_to_yuyv_round_trips_through_nchw() {
        let width = 4;
        let height = 2;
        let rgb: Vec<u8> = (0..width * height * 3).map(|i| (i * 7 % 256) as u8).collect();
        let mut yuyv = vec![0u8; width * height * 2];
        rgb_to_yuyv(&rgb, width, height, Matrix::BT601_LIMITED, &mut yuyv);
        let mut back = vec![0f32; 3 * width * height];
        yuyv_to_rgb_f32_nchw(&yuyv, width, height, Matrix::BT601_LIMITED, &mut back);
        // 4:2:2 discards half the chroma, so only luma is tightly preserved.
        for p in 0..width * height {
            let orig_y = rgb_to_yuv(rgb[p * 3], rgb[p * 3 + 1], rgb[p * 3 + 2])[0];
            let r = (back[p] * 255.0) as i32;
            let g = (back[width * height + p] * 255.0) as i32;
            let b = (back[2 * width * height + p] * 255.0) as i32;
            let got_y = rgb_to_yuv(
                r.clamp(0, 255) as u8,
                g.clamp(0, 255) as u8,
                b.clamp(0, 255) as u8,
            )[0];
            assert!((orig_y as i32 - got_y as i32).abs() <= 6, "luma drift at {p}");
        }
    }
}
