pub mod algorithms;
pub mod color_space;
pub mod color_space_lab;
pub mod enums;
pub mod error;
pub mod measured_palettes;
pub mod palettes;
pub mod tone_map;
pub mod types;

use crate::color_space::{linear_channel_to_srgb, srgb_channel_to_linear};
use crate::enums::{DitherMode, GamutCompression, ToneCompression};
use crate::palettes::Palette;
use crate::types::{AsPalette, ImageBuffer};

// ColorScheme needs both palettes and types modules, so the impl lives here.
impl AsPalette for palettes::ColorScheme {
    fn as_palette(&self) -> &palettes::Palette {
        self.palette()
    }
}

fn dispatch(data: &[u8], width: usize, height: usize, p: &Palette, mode: DitherMode, serpentine: bool) -> Vec<u8> {
    match mode {
        DitherMode::None => algorithms::direct_map(data, p),
        DitherMode::Ordered => algorithms::ordered_dither(data, width, p),
        _ => algorithms::error_diffusion_dither(data, width, height, p, mode.kernel().unwrap(), serpentine),
    }
}

/// Dither an image for e-paper display.
pub fn dither(
    img: &ImageBuffer,
    palette: impl AsPalette,
    mode: DitherMode,
    serpentine: bool,
    tone: ToneCompression,
    gamut: GamutCompression,
) -> Vec<u8> {
    let p = palette.as_palette();

    if img.data.is_empty() {
        return Vec::new();
    }

    // Fast path: no preprocessing needed
    if matches!(tone, ToneCompression::Fixed(s) if s <= 0.0)
        && matches!(gamut, GamutCompression::None)
    {
        return dispatch(img.data, img.width, img.height, p, mode, serpentine);
    }

    // Convert sRGB bytes → linear pixels, apply tone/gamut, convert back
    let mut linear: Vec<[f64; 3]> = img
        .data
        .chunks_exact(3)
        .map(|c| {
            [
                srgb_channel_to_linear(c[0]),
                srgb_channel_to_linear(c[1]),
                srgb_channel_to_linear(c[2]),
            ]
        })
        .collect();

    tone.apply(&mut linear, p);
    gamut.apply(&mut linear, p);

    let processed: Vec<u8> = linear
        .iter()
        .flat_map(|&[r, g, b]| {
            [
                linear_channel_to_srgb(r),
                linear_channel_to_srgb(g),
                linear_channel_to_srgb(b),
            ]
        })
        .collect();

    let processed_img = ImageBuffer::new(&processed, img.width);
    dispatch(processed_img.data, processed_img.width, processed_img.height, p, mode, serpentine)
}
