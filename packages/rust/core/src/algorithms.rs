//! Error diffusion and ordered dithering on raw RGB pixel buffers.

use crate::color_space::{srgb_channel_to_linear, srgb_normalized_to_linear};
use crate::color_space_lab::{PaletteLab, match_pixel, rgb_to_oklab};
use crate::palettes::Palette;
use rayon::prelude::*;

// ── Kernel definition ─────────────────────────────────────────────────────────

/// Pixel offset + pre-divided weight for one kernel entry.
pub struct KernelOffset {
    pub dx: i32,
    pub dy: i32,
    pub weight: f64,
}

pub struct Kernel {
    pub offsets: &'static [KernelOffset],
}

// ── Kernel constants ──────────────────────────────────────────────────────────

pub static FLOYD_STEINBERG: Kernel = Kernel {
    offsets: &[
        KernelOffset { dx:  1, dy: 0, weight: 7.0 / 16.0 },
        KernelOffset { dx: -1, dy: 1, weight: 3.0 / 16.0 },
        KernelOffset { dx:  0, dy: 1, weight: 5.0 / 16.0 },
        KernelOffset { dx:  1, dy: 1, weight: 1.0 / 16.0 },
    ],
};

pub static ATKINSON: Kernel = Kernel {
    offsets: &[
        KernelOffset { dx:  1, dy: 0, weight: 1.0 / 8.0 },
        KernelOffset { dx:  2, dy: 0, weight: 1.0 / 8.0 },
        KernelOffset { dx: -1, dy: 1, weight: 1.0 / 8.0 },
        KernelOffset { dx:  0, dy: 1, weight: 1.0 / 8.0 },
        KernelOffset { dx:  1, dy: 1, weight: 1.0 / 8.0 },
        KernelOffset { dx:  0, dy: 2, weight: 1.0 / 8.0 },
    ],
};

pub static BURKES: Kernel = Kernel {
    offsets: &[
        KernelOffset { dx:  1, dy: 0, weight: 8.0 / 32.0 },
        KernelOffset { dx:  2, dy: 0, weight: 4.0 / 32.0 },
        KernelOffset { dx: -2, dy: 1, weight: 2.0 / 32.0 },
        KernelOffset { dx: -1, dy: 1, weight: 4.0 / 32.0 },
        KernelOffset { dx:  0, dy: 1, weight: 8.0 / 32.0 },
        KernelOffset { dx:  1, dy: 1, weight: 4.0 / 32.0 },
        KernelOffset { dx:  2, dy: 1, weight: 2.0 / 32.0 },
    ],
};

pub static STUCKI: Kernel = Kernel {
    offsets: &[
        KernelOffset { dx:  1, dy: 0, weight: 8.0 / 42.0 },
        KernelOffset { dx:  2, dy: 0, weight: 4.0 / 42.0 },
        KernelOffset { dx: -2, dy: 1, weight: 2.0 / 42.0 },
        KernelOffset { dx: -1, dy: 1, weight: 4.0 / 42.0 },
        KernelOffset { dx:  0, dy: 1, weight: 8.0 / 42.0 },
        KernelOffset { dx:  1, dy: 1, weight: 4.0 / 42.0 },
        KernelOffset { dx:  2, dy: 1, weight: 2.0 / 42.0 },
        KernelOffset { dx: -2, dy: 2, weight: 1.0 / 42.0 },
        KernelOffset { dx: -1, dy: 2, weight: 2.0 / 42.0 },
        KernelOffset { dx:  0, dy: 2, weight: 4.0 / 42.0 },
        KernelOffset { dx:  1, dy: 2, weight: 2.0 / 42.0 },
        KernelOffset { dx:  2, dy: 2, weight: 1.0 / 42.0 },
    ],
};

pub static SIERRA: Kernel = Kernel {
    offsets: &[
        KernelOffset { dx:  1, dy: 0, weight: 5.0 / 32.0 },
        KernelOffset { dx:  2, dy: 0, weight: 3.0 / 32.0 },
        KernelOffset { dx: -2, dy: 1, weight: 2.0 / 32.0 },
        KernelOffset { dx: -1, dy: 1, weight: 4.0 / 32.0 },
        KernelOffset { dx:  0, dy: 1, weight: 5.0 / 32.0 },
        KernelOffset { dx:  1, dy: 1, weight: 4.0 / 32.0 },
        KernelOffset { dx:  2, dy: 1, weight: 2.0 / 32.0 },
        KernelOffset { dx: -1, dy: 2, weight: 2.0 / 32.0 },
        KernelOffset { dx:  0, dy: 2, weight: 3.0 / 32.0 },
        KernelOffset { dx:  1, dy: 2, weight: 2.0 / 32.0 },
    ],
};

pub static SIERRA_LITE: Kernel = Kernel {
    offsets: &[
        KernelOffset { dx:  1, dy: 0, weight: 2.0 / 4.0 },
        KernelOffset { dx: -1, dy: 1, weight: 1.0 / 4.0 },
        KernelOffset { dx:  0, dy: 1, weight: 1.0 / 4.0 },
    ],
};

pub static JARVIS_JUDICE_NINKE: Kernel = Kernel {
    offsets: &[
        KernelOffset { dx:  1, dy: 0, weight: 7.0 / 48.0 },
        KernelOffset { dx:  2, dy: 0, weight: 5.0 / 48.0 },
        KernelOffset { dx: -2, dy: 1, weight: 3.0 / 48.0 },
        KernelOffset { dx: -1, dy: 1, weight: 5.0 / 48.0 },
        KernelOffset { dx:  0, dy: 1, weight: 7.0 / 48.0 },
        KernelOffset { dx:  1, dy: 1, weight: 5.0 / 48.0 },
        KernelOffset { dx:  2, dy: 1, weight: 3.0 / 48.0 },
        KernelOffset { dx: -2, dy: 2, weight: 1.0 / 48.0 },
        KernelOffset { dx: -1, dy: 2, weight: 3.0 / 48.0 },
        KernelOffset { dx:  0, dy: 2, weight: 5.0 / 48.0 },
        KernelOffset { dx:  1, dy: 2, weight: 3.0 / 48.0 },
        KernelOffset { dx:  2, dy: 2, weight: 1.0 / 48.0 },
    ],
};

// ── Palette setup helper ─────────────────────────────────────────────────────

fn build_palette_lab(palette: &Palette) -> (Vec<[f64; 3]>, PaletteLab) {
    let palette_linear: Vec<[f64; 3]> = palette
        .colors
        .iter()
        .map(|&[r, g, b]| {
            [
                srgb_channel_to_linear(r),
                srgb_channel_to_linear(g),
                srgb_channel_to_linear(b),
            ]
        })
        .collect();
    let palette_lab = PaletteLab::from_linear_rgb(&palette_linear);
    (palette_linear, palette_lab)
}

// ── Main function ─────────────────────────────────────────────────────────────

/// Error diffusion dither. Returns palette indices (len = width × height).
pub fn error_diffusion_dither(
    pixels: &[u8],
    width: usize,
    height: usize,
    palette: &Palette,
    kernel: &Kernel,
    serpentine: bool,
) -> Vec<u8> {
    let (_palette_linear, palette_lab) = build_palette_lab(palette);

    // Palette sRGB as f64 for error computation
    let palette_srgb_f: Vec<[f64; 3]> = palette
        .colors
        .iter()
        .map(|&[r, g, b]| [r as f64, g as f64, b as f64])
        .collect();

    // Working buffer in sRGB float space [0, 255]; accumulates diffused error.
    let mut buf: Vec<f64> = pixels.iter().map(|&v| v as f64).collect();

    // LUT: u8 sRGB -> linear f64 (avoids powf per pixel in the inner loop)
    let lut: Vec<f64> = (0u8..=255).map(srgb_channel_to_linear).collect();

    let mut output = vec![0u8; width * height];

    for y in 0..height {
        // Serpentine: odd rows scan right-to-left
        let reverse = serpentine && y % 2 == 1;
        let xs: Vec<usize> = if reverse {
            (0..width).rev().collect()
        } else {
            (0..width).collect()
        };

        for x in xs {
            let idx = (y * width + x) * 3;

            let rs = buf[idx].clamp(0.0, 255.0);
            let gs = buf[idx + 1].clamp(0.0, 255.0);
            let bs = buf[idx + 2].clamp(0.0, 255.0);

            let r_lin = lut[rs.round() as usize];
            let g_lin = lut[gs.round() as usize];
            let b_lin = lut[bs.round() as usize];

            let pixel_lab = rgb_to_oklab(r_lin, g_lin, b_lin);
            let best_idx = match_pixel(pixel_lab, &palette_lab);

            output[y * width + x] = best_idx as u8;

            // Quantization error in sRGB space
            let err_r = rs - palette_srgb_f[best_idx][0];
            let err_g = gs - palette_srgb_f[best_idx][1];
            let err_b = bs - palette_srgb_f[best_idx][2];

            for offset in kernel.offsets {
                let effective_dx = if reverse { -offset.dx } else { offset.dx };

                let nx = x as i64 + effective_dx as i64;
                let ny = y as i64 + offset.dy as i64;

                if nx >= 0 && nx < width as i64 && ny >= 0 && ny < height as i64 {
                    let ni = (ny as usize * width + nx as usize) * 3;
                    buf[ni] += err_r * offset.weight;
                    buf[ni + 1] += err_g * offset.weight;
                    buf[ni + 2] += err_b * offset.weight;
                }
            }
        }
    }

    output
}

// ── Thin wrappers ─────────────────────────────────────────────────────────────

pub fn floyd_steinberg(pixels: &[u8], w: usize, h: usize, palette: &Palette, serpentine: bool) -> Vec<u8> {
    error_diffusion_dither(pixels, w, h, palette, &FLOYD_STEINBERG, serpentine)
}
pub fn atkinson(pixels: &[u8], w: usize, h: usize, palette: &Palette, serpentine: bool) -> Vec<u8> {
    error_diffusion_dither(pixels, w, h, palette, &ATKINSON, serpentine)
}
pub fn burkes(pixels: &[u8], w: usize, h: usize, palette: &Palette, serpentine: bool) -> Vec<u8> {
    error_diffusion_dither(pixels, w, h, palette, &BURKES, serpentine)
}
pub fn stucki(pixels: &[u8], w: usize, h: usize, palette: &Palette, serpentine: bool) -> Vec<u8> {
    error_diffusion_dither(pixels, w, h, palette, &STUCKI, serpentine)
}
pub fn sierra(pixels: &[u8], w: usize, h: usize, palette: &Palette, serpentine: bool) -> Vec<u8> {
    error_diffusion_dither(pixels, w, h, palette, &SIERRA, serpentine)
}
pub fn sierra_lite(pixels: &[u8], w: usize, h: usize, palette: &Palette, serpentine: bool) -> Vec<u8> {
    error_diffusion_dither(pixels, w, h, palette, &SIERRA_LITE, serpentine)
}
pub fn jarvis_judice_ninke(pixels: &[u8], w: usize, h: usize, palette: &Palette, serpentine: bool) -> Vec<u8> {
    error_diffusion_dither(pixels, w, h, palette, &JARVIS_JUDICE_NINKE, serpentine)
}

// ── Direct palette map (no dithering) ────────────────────────────────────────

/// Nearest-color mapping with no dithering. Each pixel maps independently.
pub fn direct_map(pixels: &[u8], palette: &Palette) -> Vec<u8> {
    let (_, palette_lab) = build_palette_lab(palette);
    pixels
        .par_chunks(3)
        .map(|rgb| {
            let r = srgb_channel_to_linear(rgb[0]);
            let g = srgb_channel_to_linear(rgb[1]);
            let b = srgb_channel_to_linear(rgb[2]);
            let lab = rgb_to_oklab(r, g, b);
            match_pixel(lab, &palette_lab) as u8
        })
        .collect()
}

// ── Ordered (Bayer) dithering ─────────────────────────────────────────────────

// 4×4 Bayer matrix, normalized to [-0.5, 0.5]. Indexed as [y % 4][x % 4].
// Values = (bayer_entry / 16.0) - 0.5 for the standard ordered Bayer matrix.
// Written as exact fractions to keep precision (all are multiples of 1/16).
const BAYER_4X4: [[f64; 4]; 4] = [
    [-0.5000,  0.0000, -0.3750,  0.1250],
    [ 0.2500, -0.2500,  0.3750, -0.1250],
    [-0.3125,  0.1875, -0.4375,  0.0625],
    [ 0.4375, -0.0625,  0.3125, -0.1875],
];

/// Ordered (Bayer 4×4) dither. Pixels are independent — parallelized with rayon.
pub fn ordered_dither(pixels: &[u8], width: usize, palette: &Palette) -> Vec<u8> {
    let (_palette_linear, palette_lab) = build_palette_lab(palette);

    pixels
        .par_chunks(3)
        .enumerate()
        .map(|(i, rgb)| {
            let x = i % width;
            let y = i / width;

            let threshold = BAYER_4X4[y % 4][x % 4];

            // Apply threshold in perceptual (sRGB) space for uniform dithering
            // across the tonal range, then convert to linear for OKLab matching.
            let r = srgb_normalized_to_linear((rgb[0] as f64 / 255.0 + threshold).clamp(0.0, 1.0));
            let g = srgb_normalized_to_linear((rgb[1] as f64 / 255.0 + threshold).clamp(0.0, 1.0));
            let b = srgb_normalized_to_linear((rgb[2] as f64 / 255.0 + threshold).clamp(0.0, 1.0));

            let lab = rgb_to_oklab(r, g, b);
            match_pixel(lab, &palette_lab) as u8
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::palettes::ColorScheme;

    fn solid_image(r: u8, g: u8, b: u8, w: usize, h: usize) -> Vec<u8> {
        vec![r, g, b].into_iter().cycle().take(w * h * 3).collect()
    }

    #[test]
    fn solid_white_maps_to_white() {
        let pixels = solid_image(255, 255, 255, 4, 4);
        let out = floyd_steinberg(&pixels, 4, 4, ColorScheme::Mono.palette(), true);
        // White palette index in Mono is 1
        assert!(out.iter().all(|&v| v == 1));
    }

    #[test]
    fn solid_black_maps_to_black() {
        let pixels = solid_image(0, 0, 0, 4, 4);
        let out = floyd_steinberg(&pixels, 4, 4, ColorScheme::Mono.palette(), true);
        assert!(out.iter().all(|&v| v == 0));
    }

    #[test]
    fn output_length_matches_pixels() {
        let pixels = solid_image(128, 64, 200, 10, 7);
        let out = floyd_steinberg(&pixels, 10, 7, ColorScheme::Bwr.palette(), true);
        assert_eq!(out.len(), 10 * 7);
    }

    #[test]
    fn all_algorithms_produce_correct_length() {
        let pixels = solid_image(128, 64, 200, 10, 7);
        let pal = ColorScheme::Bwr.palette();
        assert_eq!(burkes(&pixels, 10, 7, pal, false).len(), 70);
        assert_eq!(atkinson(&pixels, 10, 7, pal, false).len(), 70);
        assert_eq!(stucki(&pixels, 10, 7, pal, false).len(), 70);
        assert_eq!(sierra(&pixels, 10, 7, pal, false).len(), 70);
        assert_eq!(sierra_lite(&pixels, 10, 7, pal, false).len(), 70);
        assert_eq!(jarvis_judice_ninke(&pixels, 10, 7, pal, false).len(), 70);
        assert_eq!(ordered_dither(&pixels, 10, pal).len(), 70);
        assert_eq!(direct_map(&pixels, pal).len(), 70);
    }

    #[test]
    fn direct_map_pure_red_maps_to_red_in_bwr() {
        // BWR palette: index 0=black, 1=white, 2=red — pure red should map to red ink
        let pixels = vec![255u8, 0, 0];
        let out = direct_map(&pixels, ColorScheme::Bwr.palette());
        assert_eq!(out[0], 2, "pure sRGB red should map to red ink (index 2) in BWR");
    }

    #[test]
    fn direct_map_pure_blue_maps_to_blue_in_bwgbry() {
        // BWGBRY palette order: 0=black, 1=white, 2=yellow, 3=red, 4=blue, 5=green
        let pixels = vec![0u8, 0, 255];
        let out = direct_map(&pixels, ColorScheme::Bwgbry.palette());
        assert_eq!(out[0], 4, "pure sRGB blue should map to blue ink (index 4) in BWGBRY");
    }

    #[test]
    fn serpentine_differs_from_raster_on_gradient() {
        // A gradient (non-uniform content) should produce different output
        // with serpentine=true vs serpentine=false
        let pixels: Vec<u8> = (0u8..=255)
            .flat_map(|v| [v, v / 2, 255 - v])
            .cycle()
            .take(16 * 16 * 3)
            .collect();
        let raster = floyd_steinberg(&pixels, 16, 16, ColorScheme::Mono.palette(), false);
        let serpentine = floyd_steinberg(&pixels, 16, 16, ColorScheme::Mono.palette(), true);
        assert_ne!(raster, serpentine, "serpentine and raster should differ on a gradient");
    }

    #[test]
    fn serpentine_first_row_matches_raster() {
        // Row 0 is always scanned left-to-right regardless of serpentine flag
        let pixels: Vec<u8> = (0u8..=255)
            .flat_map(|v| [v, 128u8.wrapping_add(v), 64u8])
            .cycle()
            .take(8 * 4 * 3)
            .collect();
        let raster = floyd_steinberg(&pixels, 8, 4, ColorScheme::Mono.palette(), false);
        let serpentine = floyd_steinberg(&pixels, 8, 4, ColorScheme::Mono.palette(), true);
        assert_eq!(
            &raster[0..8],
            &serpentine[0..8],
            "first row must be identical (both scan left-to-right)"
        );
    }
}
