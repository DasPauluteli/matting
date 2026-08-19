//! YUYV 4:2:2 <-> RGB conversion.
//!
//! Uses the BT.601 limited-range matrix, which is what USB and loopback
//! webcams overwhelmingly produce. If colours look washed out or over-saturated
//! against the real camera, this is the first constant to revisit.

#[inline]
pub fn yuv_to_rgb(y: u8, u: u8, v: u8) -> [u8; 3] {
    let y = (y as f32 - 16.0) * 1.164_383;
    let u = u as f32 - 128.0;
    let v = v as f32 - 128.0;
    let r = y + 1.596_027 * v;
    let g = y - 0.391_762 * u - 0.812_968 * v;
    let b = y + 2.017_232 * u;
    [clamp_u8(r), clamp_u8(g), clamp_u8(b)]
}

#[inline]
pub fn rgb_to_yuv(r: u8, g: u8, b: u8) -> [u8; 3] {
    let (r, g, b) = (r as f32, g as f32, b as f32);
    let y = 0.256_788 * r + 0.504_129 * g + 0.097_906 * b + 16.0;
    let u = -0.148_223 * r - 0.290_993 * g + 0.439_216 * b + 128.0;
    let v = 0.439_216 * r - 0.367_788 * g - 0.071_427 * b + 128.0;
    [clamp_u8(y), clamp_u8(u), clamp_u8(v)]
}

#[inline]
fn clamp_u8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

/// YUYV 4:2:2 -> planar RGB, normalized to [0,1], laid out NCHW as the model
/// expects. `out` must be `3 * width * height` long.
pub fn yuyv_to_rgb_f32_nchw(yuyv: &[u8], width: usize, height: usize, out: &mut [f32]) {
    let plane = width * height;
    debug_assert_eq!(out.len(), plane * 3);
    debug_assert_eq!(yuyv.len(), plane * 2);

    for row in 0..height {
        let src = row * width * 2;
        let dst = row * width;
        for pair in 0..width / 2 {
            let i = src + pair * 4;
            let (y0, u, y1, v) = (yuyv[i], yuyv[i + 1], yuyv[i + 2], yuyv[i + 3]);
            let p0 = yuv_to_rgb(y0, u, v);
            let p1 = yuv_to_rgb(y1, u, v);
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
pub fn rgb_to_yuyv(rgb: &[u8], width: usize, height: usize, out: &mut [u8]) {
    debug_assert_eq!(rgb.len(), width * height * 3);
    debug_assert_eq!(out.len(), width * height * 2);

    for row in 0..height {
        let src = row * width * 3;
        let dst = row * width * 2;
        for pair in 0..width / 2 {
            let a = src + pair * 6;
            let p0 = rgb_to_yuv(rgb[a], rgb[a + 1], rgb[a + 2]);
            let p1 = rgb_to_yuv(rgb[a + 3], rgb[a + 4], rgb[a + 5]);
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
        yuyv_to_rgb_f32_nchw(&yuyv, 2, 1, &mut out);
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
        rgb_to_yuyv(&rgb, width, height, &mut yuyv);
        let mut back = vec![0f32; 3 * width * height];
        yuyv_to_rgb_f32_nchw(&yuyv, width, height, &mut back);
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
