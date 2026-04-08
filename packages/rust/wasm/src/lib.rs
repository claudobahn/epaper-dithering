use epaper_dithering_core::enums::{DitherMode, GamutCompression, Sharpening, ToneCompression};
use epaper_dithering_core::palettes::{ColorScheme, Palette};
use epaper_dithering_core::types::ImageBuffer;
use epaper_dithering_core::dither;
use wasm_bindgen::prelude::*;

fn parse_scheme(v: u8) -> Result<ColorScheme, JsValue> {
    ColorScheme::try_from(v).map_err(|e| JsValue::from_str(&e.to_string()))
}

fn parse_mode(v: u8) -> Result<DitherMode, JsValue> {
    DitherMode::try_from(v).map_err(|e| JsValue::from_str(&e.to_string()))
}

fn parse_tone(v: Option<f64>) -> ToneCompression {
    match v {
        None => ToneCompression::Auto,
        Some(s) => ToneCompression::Fixed(s),
    }
}

fn parse_gamut(v: Option<f64>) -> GamutCompression {
    match v {
        None => GamutCompression::Auto,
        Some(s) => GamutCompression::Fixed(s),
    }
}

/// Dither a flat RGB image for an idealized e-paper color scheme.
///
/// - `pixels`: flat RGB bytes, row-major (len = width × height × 3)
/// - `scheme_id`: firmware color scheme (0=mono … 7=grayscale16)
/// - `mode_id`: dither algorithm (0=none … 8=jjn)
/// - `tone_compression`: ignored for idealized palettes — pass 0.0
/// - `gamut_compression`: 0.0 = off, 0.0–1.0 = fixed strength
///
/// Returns a Uint8Array of palette indices (one per pixel).
#[wasm_bindgen]
pub fn dither_image(
    pixels: &[u8],
    width: usize,
    scheme_id: u8,
    mode_id: u8,
    serpentine: bool,
    tone_compression: Option<f64>,
    gamut_compression: Option<f64>,
) -> Result<Vec<u8>, JsValue> {
    let palette = parse_scheme(scheme_id)?.palette();
    let img = ImageBuffer::new(pixels, width);
    Ok(dither(
        &img,
        palette,
        parse_mode(mode_id)?,
        serpentine,
        parse_tone(tone_compression),
        parse_gamut(gamut_compression),
        Sharpening::None,
    ))
}

/// Dither a flat RGB image using a measured ColorPalette.
///
/// - `pixels`: flat RGB bytes, row-major (len = width × height × 3)
/// - `palette_bytes`: flat RGB bytes for each palette color (len = n_colors × 3)
/// - `accent_idx`: index of the accent color in the palette
/// - `mode_id`: dither algorithm (0=none … 8=jjn)
/// - `tone_compression`: null = auto, 0.0 = off, 0.0–1.0 = fixed
/// - `gamut_compression`: null = auto, 0.0 = off, 0.0–1.0 = fixed
///
/// Returns a Uint8Array of palette indices (one per pixel).
#[wasm_bindgen]
pub fn dither_image_palette(
    pixels: &[u8],
    width: usize,
    palette_bytes: &[u8],
    accent_idx: usize,
    mode_id: u8,
    serpentine: bool,
    tone_compression: Option<f64>,
    gamut_compression: Option<f64>,
) -> Result<Vec<u8>, JsValue> {
    let colors: Vec<[u8; 3]> = palette_bytes
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let palette = Palette::new(colors, accent_idx);
    let img = ImageBuffer::new(pixels, width);
    Ok(dither(
        &img,
        palette,
        parse_mode(mode_id)?,
        serpentine,
        parse_tone(tone_compression),
        parse_gamut(gamut_compression),
        Sharpening::None,
    ))
}
