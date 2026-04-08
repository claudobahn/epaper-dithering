use crate::algorithms::{
    Kernel, ATKINSON, BURKES, FLOYD_STEINBERG, JARVIS_JUDICE_NINKE, SIERRA, SIERRA_LITE, STUCKI,
};
use crate::error::DitherError;
use crate::palettes::Palette;
use crate::tone_map;

/// Dithering algorithm. Integer values match OpenDisplay firmware conventions.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DitherMode {
    None           = 0,
    #[default]
    Burkes         = 1,
    Ordered        = 2,
    FloydSteinberg = 3,
    Atkinson       = 4,
    Stucki         = 5,
    Sierra         = 6,
    SierraLite     = 7,
    JarvisJudiceNinke = 8,
}

/// Dynamic range compression applied before dithering.
/// Only meaningful for measured palettes (ignored for `ColorScheme`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ToneCompression {
    /// Reinhard 2004 skewness-based auto strength. Best for photos.
    #[default]
    Auto,
    /// Fixed blend strength in [0.0, 1.0]. 0.0 disables.
    Fixed(f64),
}

impl ToneCompression {
    pub fn apply(self, pixels: &mut [[f64; 3]], palette: &Palette) {
        match self {
            ToneCompression::Auto => tone_map::auto_compress_dynamic_range(pixels, palette),
            ToneCompression::Fixed(s) => if s > 0.0 {
                tone_map::compress_dynamic_range(pixels, palette, s)
            }
        }
    }
}

/// Gamut compression applied before dithering.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum GamutCompression {
    /// No gamut compression (default).
    #[default]
    None,
    /// Full gamut compression at strength 1.0.
    Auto,
    /// Fixed blend strength in [0.0, 1.0].
    Fixed(f64),
}

impl GamutCompression {
    pub fn apply(self, pixels: &mut [[f64; 3]], palette: &Palette) {
        match self {
            GamutCompression::None => {}
            GamutCompression::Auto => tone_map::auto_gamut_compress(pixels, palette),
            GamutCompression::Fixed(s) => if s > 0.0 {
                tone_map::gamut_compress(pixels, palette, s)
            }
        }
    }


}

/// Pre-dithering sharpening to counteract error-diffusion blur.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Sharpening {
    /// No sharpening (default).
    #[default]
    None,
    /// Unsharp mask with the given amount (typical: 0.5–1.5) and radius 1.
    Unsharp(f64),
}

impl Sharpening {
    pub fn apply(self, pixels: &mut [[f64; 3]], width: usize, height: usize) {
        match self {
            Sharpening::None => {}
            Sharpening::Unsharp(amount) => if amount > 0.0 {
                crate::sharpen::unsharp_mask(pixels, width, height, amount, 1)
            }
        }
    }
}

impl TryFrom<u8> for DitherMode {
    type Error = DitherError;

    fn try_from(v: u8) -> Result<Self, Self::Error> {
        match v {
            0 => Ok(DitherMode::None),
            1 => Ok(DitherMode::Burkes),
            2 => Ok(DitherMode::Ordered),
            3 => Ok(DitherMode::FloydSteinberg),
            4 => Ok(DitherMode::Atkinson),
            5 => Ok(DitherMode::Stucki),
            6 => Ok(DitherMode::Sierra),
            7 => Ok(DitherMode::SierraLite),
            8 => Ok(DitherMode::JarvisJudiceNinke),
            _ => Err(DitherError::UnknownDitherMode(v)),
        }
    }
}

impl DitherMode {
    pub fn kernel(self) -> Option<&'static Kernel> {
        match self {
            DitherMode::None | DitherMode::Ordered => None,
            DitherMode::FloydSteinberg => Some(&FLOYD_STEINBERG),
            DitherMode::Burkes => Some(&BURKES),
            DitherMode::Atkinson => Some(&ATKINSON),
            DitherMode::Stucki => Some(&STUCKI),
            DitherMode::Sierra => Some(&SIERRA),
            DitherMode::SierraLite => Some(&SIERRA_LITE),
            DitherMode::JarvisJudiceNinke => Some(&JARVIS_JUDICE_NINKE),
        }
    }
}
