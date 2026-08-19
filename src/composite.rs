use anyhow::{Context, Result};
use image::imageops::FilterType;

use crate::cli::Fit;

#[inline]
fn to_u8(v: f32) -> u8 {
    (v * 255.0).round().clamp(0.0, 255.0) as u8
}

/// Planar float foreground + alpha -> interleaved BGRA8, preserving alpha.
/// This is the only mode that carries true transparency.
///
/// RVM's foreground prediction is only meaningful where alpha is non-zero; in
/// the background it holds a smeared inpainting of the scene. A compositor that
/// treats the stream as premultiplied (OBS does) would draw that garbage over
/// the background, so premultiplying is the default: it forces transparent
/// pixels to black and makes the output well-defined either way.
pub fn to_bgra(fgr: &[f32], pha: &[f32], plane: usize, premultiplied: bool, out: &mut [u8]) {
    debug_assert_eq!(out.len(), plane * 4);
    for i in 0..plane {
        let a = pha[i];
        let scale = if premultiplied { a } else { 1.0 };
        out[i * 4] = to_u8(fgr[2 * plane + i] * scale);
        out[i * 4 + 1] = to_u8(fgr[plane + i] * scale);
        out[i * 4 + 2] = to_u8(fgr[i] * scale);
        out[i * 4 + 3] = to_u8(a);
    }
}

/// `out = fgr*pha + colour*(1-pha)`, interleaved RGB8.
pub fn over_color(fgr: &[f32], pha: &[f32], plane: usize, colour: [u8; 3], out: &mut [u8]) {
    debug_assert_eq!(out.len(), plane * 3);
    for i in 0..plane {
        let a = pha[i];
        for c in 0..3 {
            let f = fgr[c * plane + i];
            out[i * 3 + c] = to_u8(f * a + (colour[c] as f32 / 255.0) * (1.0 - a));
        }
    }
}

/// `out = fgr*pha + bg*(1-pha)`, interleaved RGB8. `bg` is interleaved RGB8.
pub fn over_image(fgr: &[f32], pha: &[f32], plane: usize, bg: &[u8], out: &mut [u8]) {
    debug_assert_eq!(out.len(), plane * 3);
    debug_assert_eq!(bg.len(), plane * 3);
    for i in 0..plane {
        let a = pha[i];
        for c in 0..3 {
            let f = fgr[c * plane + i];
            out[i * 3 + c] = to_u8(f * a + (bg[i * 3 + c] as f32 / 255.0) * (1.0 - a));
        }
    }
}

/// A background image pre-resized to the capture resolution, so the per-frame
/// cost is only the blend.
pub struct Background {
    rgb: Vec<u8>,
}

impl Background {
    pub fn load(path: &str, width: u32, height: u32, fit: Fit) -> Result<Background> {
        let img = image::open(path)
            .with_context(|| format!("opening background {path}"))?
            .to_rgb8();
        let out = match fit {
            Fit::Stretch => image::imageops::resize(&img, width, height, FilterType::CatmullRom),
            Fit::Cover => {
                let scale =
                    (width as f32 / img.width() as f32).max(height as f32 / img.height() as f32);
                let (sw, sh) = (
                    (img.width() as f32 * scale).ceil() as u32,
                    (img.height() as f32 * scale).ceil() as u32,
                );
                let scaled = image::imageops::resize(&img, sw, sh, FilterType::CatmullRom);
                image::imageops::crop_imm(&scaled, (sw - width) / 2, (sh - height) / 2, width, height)
                    .to_image()
            }
            Fit::Contain => {
                let scale =
                    (width as f32 / img.width() as f32).min(height as f32 / img.height() as f32);
                let (sw, sh) = (
                    ((img.width() as f32 * scale) as u32).max(1),
                    ((img.height() as f32 * scale) as u32).max(1),
                );
                let scaled = image::imageops::resize(&img, sw, sh, FilterType::CatmullRom);
                let mut canvas = image::RgbImage::from_pixel(width, height, image::Rgb([0, 0, 0]));
                image::imageops::overlay(
                    &mut canvas,
                    &scaled,
                    ((width - sw) / 2) as i64,
                    ((height - sh) / 2) as i64,
                );
                canvas
            }
        };
        Ok(Background {
            rgb: out.into_raw(),
        })
    }

    pub fn rgb(&self) -> &[u8] {
        &self.rgb
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // One pixel, fully opaque red foreground.
    fn red_fg() -> (Vec<f32>, usize) {
        (vec![1.0, 0.0, 0.0], 1)
    }

    #[test]
    fn alpha_mode_writes_bgra_with_alpha() {
        let (fgr, plane) = red_fg();
        let mut out = vec![0u8; 4];
        to_bgra(&fgr, &[1.0], plane, false, &mut out);
        assert_eq!(out, vec![0, 0, 255, 255], "expected B,G,R,A");

        to_bgra(&fgr, &[0.0], plane, false, &mut out);
        assert_eq!(out[3], 0, "transparent pixel must have alpha 0");
    }

    /// RVM leaves garbage in `fgr` where alpha is zero. Premultiplying must
    /// force those pixels to black so a premultiplied compositor (OBS) does not
    /// draw the garbage over the background.
    #[test]
    fn premultiplying_blacks_out_transparent_garbage() {
        let garbage = vec![0.9f32, 0.4, 0.7];
        let mut out = vec![0u8; 4];
        to_bgra(&garbage, &[0.0], 1, true, &mut out);
        assert_eq!(out, vec![0, 0, 0, 0], "transparent pixels must be fully black");

        // Straight alpha keeps the garbage, which is what OBS was showing.
        to_bgra(&garbage, &[0.0], 1, false, &mut out);
        assert_ne!(&out[..3], &[0, 0, 0], "straight alpha should preserve colour");
    }

    #[test]
    fn premultiplying_scales_partial_alpha() {
        let fgr = vec![1.0f32, 1.0, 1.0];
        let mut out = vec![0u8; 4];
        to_bgra(&fgr, &[0.5], 1, true, &mut out);
        assert!((out[0] as i32 - 128).abs() <= 1, "b={}", out[0]);
        assert_eq!(out[3], 128, "alpha itself must not be scaled");
    }

    #[test]
    fn greenscreen_shows_key_colour_where_transparent() {
        let (fgr, plane) = red_fg();
        let mut out = vec![0u8; 3];
        over_color(&fgr, &[0.0], plane, [0, 255, 0], &mut out);
        assert_eq!(out, vec![0, 255, 0], "fully transparent should be pure key");

        over_color(&fgr, &[1.0], plane, [0, 255, 0], &mut out);
        assert_eq!(out, vec![255, 0, 0], "fully opaque should be pure foreground");
    }

    #[test]
    fn greenscreen_blends_at_half_alpha() {
        let (fgr, plane) = red_fg();
        let mut out = vec![0u8; 3];
        over_color(&fgr, &[0.5], plane, [0, 255, 0], &mut out);
        assert!((out[0] as i32 - 128).abs() <= 1, "r={}", out[0]);
        assert!((out[1] as i32 - 128).abs() <= 1, "g={}", out[1]);
        assert_eq!(out[2], 0);
    }

    #[test]
    fn image_mode_shows_background_where_transparent() {
        let (fgr, plane) = red_fg();
        let bg = [10u8, 20, 30];
        let mut out = vec![0u8; 3];
        over_image(&fgr, &[0.0], plane, &bg, &mut out);
        assert_eq!(out, vec![10, 20, 30]);

        over_image(&fgr, &[1.0], plane, &bg, &mut out);
        assert_eq!(out, vec![255, 0, 0]);
    }

    #[test]
    fn cover_fit_matches_target_dimensions() {
        let dir = std::env::temp_dir().join(format!("matting-bg-{}.png", std::process::id()));
        image::RgbImage::from_pixel(64, 64, image::Rgb([1, 2, 3]))
            .save(&dir)
            .unwrap();
        let bg = Background::load(dir.to_str().unwrap(), 1024, 576, crate::cli::Fit::Cover).unwrap();
        assert_eq!(bg.rgb().len(), 1024 * 576 * 3);
        std::fs::remove_file(&dir).ok();
    }

    #[test]
    fn contain_fit_matches_target_dimensions() {
        let dir = std::env::temp_dir().join(format!("matting-bg2-{}.png", std::process::id()));
        image::RgbImage::from_pixel(64, 32, image::Rgb([4, 5, 6]))
            .save(&dir)
            .unwrap();
        let bg =
            Background::load(dir.to_str().unwrap(), 1024, 576, crate::cli::Fit::Contain).unwrap();
        assert_eq!(bg.rgb().len(), 1024 * 576 * 3);
        std::fs::remove_file(&dir).ok();
    }
}
