use epaper_dithering_core::dither;
use epaper_dithering_core::enums::{DitherMode, GamutCompression, Sharpening, ToneCompression};
use epaper_dithering_core::palettes::{ColorScheme, Palette};
use epaper_dithering_core::types::ImageBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn parse_scheme(v: u8) -> PyResult<ColorScheme> {
    ColorScheme::try_from(v).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn parse_mode(v: u8) -> PyResult<DitherMode> {
    DitherMode::try_from(v).map_err(|e| PyValueError::new_err(e.to_string()))
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
        Some(s) if s > 0.0 => GamutCompression::Fixed(s),
        _ => GamutCompression::None,
    }
}

#[pyfunction]
#[pyo3(signature = (pixels, width, height, scheme_id, mode_id=3, serpentine=true, tone_compression=None, gamut_compression=None))]
fn dither_image(
    pixels: &[u8],
    width: usize,
    height: usize,
    scheme_id: u8,
    mode_id: u8,
    serpentine: bool,
    tone_compression: Option<f64>,
    gamut_compression: Option<f64>,
) -> PyResult<Vec<u8>> {
    let palette = parse_scheme(scheme_id)?.palette();
    let img = ImageBuffer::new(pixels, width);
    let _ = height;
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

#[pyfunction]
#[pyo3(signature = (pixels, width, height, palette_bytes, accent_idx=0, mode_id=3, serpentine=true, tone_compression=None, gamut_compression=None))]
fn dither_image_palette(
    pixels: &[u8],
    width: usize,
    height: usize,
    palette_bytes: &[u8],
    accent_idx: usize,
    mode_id: u8,
    serpentine: bool,
    tone_compression: Option<f64>,
    gamut_compression: Option<f64>,
) -> PyResult<Vec<u8>> {
    if palette_bytes.len() % 3 != 0 {
        return Err(PyValueError::new_err("palette_bytes length must be a multiple of 3"));
    }
    let colors: Vec<[u8; 3]> = palette_bytes
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let palette = Palette::new(colors, accent_idx);
    let img = ImageBuffer::new(pixels, width);
    let _ = height;
    Ok(dither(
        &img,
        &palette,
        parse_mode(mode_id)?,
        serpentine,
        parse_tone(tone_compression),
        parse_gamut(gamut_compression),
        Sharpening::None,
    ))
}

#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dither_image, m)?)?;
    m.add_function(wrap_pyfunction!(dither_image_palette, m)?)?;
    Ok(())
}
