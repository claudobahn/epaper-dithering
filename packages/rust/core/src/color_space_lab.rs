//! OKLab color space and weighted Cartesian color matching.
//!
//! Uses weighted Euclidean distance in OKLab (L, a, b) with a modest boost to
//! chrominance axes (Wab=1.5) to prioritise color-ink usage on e-paper displays
//! while keeping lightness accuracy for structural detail.

// sRGB -> XYZ matrix (D65 illuminant, BruceLinbloom)
const M_RGB_XYZ: [[f64; 3]; 3] = [
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
];

// XYZ -> LMS (M1, Ottosson)
const M1: [[f64; 3]; 3] = [
    [0.8189330101, 0.3618667424, -0.1288597137],
    [0.0329845436, 0.9293118715, 0.0361456387],
    [0.0482003018, 0.2643662691, 0.6338517070],
];

// cbrt(LMS) -> OKLab (M2, Ottosson)
const M2: [[f64; 3]; 3] = [
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
];

// Weighted Cartesian OKLab distance weights.
// Wab > 1 slightly favours chromatic palette entries over achromatic ones,
// which helps e-paper displays utilise their limited colour inks.
const WL: f64 = 1.0;
const WAB: f64 = 1.5;

#[derive(Debug, Clone, Copy)]
pub struct OkLab {
    pub l: f64,
    pub a: f64,
    pub b: f64,
}

impl OkLab {
    pub fn chroma(&self) -> f64 {
        (self.a * self.a + self.b * self.b).sqrt()
    }
}

/// Linear RGB → OKLab. Pipeline: RGB → XYZ → LMS → cbrt → OKLab.
pub fn rgb_to_oklab(r: f64, g: f64, b: f64) -> OkLab {
    let x  = M_RGB_XYZ[0][0] * r + M_RGB_XYZ[0][1] * g + M_RGB_XYZ[0][2] * b;
    let y  = M_RGB_XYZ[1][0] * r + M_RGB_XYZ[1][1] * g + M_RGB_XYZ[1][2] * b;
    let z  = M_RGB_XYZ[2][0] * r + M_RGB_XYZ[2][1] * g + M_RGB_XYZ[2][2] * b;

    let l = M1[0][0] * x + M1[0][1] * y + M1[0][2] * z;
    let m = M1[1][0] * x + M1[1][1] * y + M1[1][2] * z;
    let s = M1[2][0] * x + M1[2][1] * y + M1[2][2] * z;

    let l_ = l.cbrt();
    let m_ = m.cbrt();
    let s_ = s.cbrt();

    OkLab {
        l: M2[0][0] * l_ + M2[0][1] * m_ + M2[0][2] * s_,
        a: M2[1][0] * l_ + M2[1][1] * m_ + M2[1][2] * s_,
        b: M2[2][0] * l_ + M2[2][1] * m_ + M2[2][2] * s_,
    }
}

pub struct PaletteLab {
    pub colors: Vec<OkLab>,
}

impl PaletteLab {
    pub fn from_linear_rgb(palette: &[[f64; 3]]) -> Self {
        let colors: Vec<OkLab> = palette
            .iter()
            .map(|c| rgb_to_oklab(c[0], c[1], c[2]))
            .collect();
        Self { colors }
    }
}

/// Returns the index of the closest palette color (weighted Cartesian OKLab distance).
pub fn match_pixel(pixel: OkLab, palette: &PaletteLab) -> usize {
    let mut best_idx = 0;
    let mut best_dist = f64::INFINITY;

    for (i, pal) in palette.colors.iter().enumerate() {
        let dl = pixel.l - pal.l;
        let da = pixel.a - pal.a;
        let db = pixel.b - pal.b;

        let dist = (WL * dl) * (WL * dl)
            + (WAB * da) * (WAB * da)
            + (WAB * db) * (WAB * db);
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
    }

    best_idx
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn black_is_zero() {
        let lab = rgb_to_oklab(0.0, 0.0, 0.0);
        assert_relative_eq!(lab.l, 0.0, epsilon = 1e-6);
        assert_relative_eq!(lab.a, 0.0, epsilon = 1e-6);
        assert_relative_eq!(lab.b, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn white_l_is_one() {
        let lab = rgb_to_oklab(1.0, 1.0, 1.0);
        assert_relative_eq!(lab.l, 1.0, epsilon = 1e-4);
        assert_relative_eq!(lab.a, 0.0, epsilon = 1e-4);
        assert_relative_eq!(lab.b, 0.0, epsilon = 1e-4);
    }

    #[test]
    fn match_exact_palette_color() {
        let red_linear = [0.2126, 0.0, 0.0_f64];
        let palette = PaletteLab::from_linear_rgb(&[red_linear, [0.0, 0.7152, 0.0]]);
        let pixel = rgb_to_oklab(red_linear[0], red_linear[1], red_linear[2]);
        assert_eq!(match_pixel(pixel, &palette), 0);
    }

    #[test]
    fn achromatic_does_not_attract_saturated_pixels() {
        // A saturated pixel should not preferentially match an achromatic palette
        // entry just because hue is undefined at zero chroma.
        // Vivid blue should match a blue-ish palette entry, not black.
        let target = rgb_to_oklab(0.0, 0.0, 0.8); // vivid blue
        let blue_ish = [0.0_f64, 0.05, 0.4];       // dark blue
        let black    = [0.0_f64, 0.0,  0.0];        // achromatic

        let palette = PaletteLab::from_linear_rgb(&[blue_ish, black]);
        let result = match_pixel(target, &palette);
        assert_eq!(
            result, 0,
            "saturated blue should match dark blue, not achromatic black"
        );
    }
}
