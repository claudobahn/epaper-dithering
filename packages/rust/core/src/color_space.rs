/// sRGB [0–255] → linear [0.0–1.0].
pub fn srgb_channel_to_linear(value: u8) -> f64 {
    srgb_normalized_to_linear(value as f64 / 255.0)
}

/// Normalized sRGB [0.0–1.0] → linear [0.0–1.0]. IEC 61966-2-1 piecewise transfer function.
pub fn srgb_normalized_to_linear(value: f64) -> f64 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear [0.0–1.0] → sRGB [0–255]. Inverse of `srgb_channel_to_linear`.
pub fn linear_channel_to_srgb(linear: f64) -> u8 {
    let srgb = if linear <= 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0).round() as u8
}

pub fn srgb_to_linear(r: u8, g: u8, b: u8) -> (f64, f64, f64) {
    (
        srgb_channel_to_linear(r),
        srgb_channel_to_linear(g),
        srgb_channel_to_linear(b),
    )
}

pub fn linear_to_srgb(r: f64, g: f64, b: f64) -> (u8, u8, u8) {
    (
        linear_channel_to_srgb(r),
        linear_channel_to_srgb(g),
        linear_channel_to_srgb(b),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn black_and_white_are_identity() {
        assert_relative_eq!(srgb_channel_to_linear(0), 0.0);
        assert_relative_eq!(srgb_channel_to_linear(255), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn roundtrip() {
        for v in [0u8, 1, 10, 127, 128, 254, 255] {
            assert_eq!(linear_channel_to_srgb(srgb_channel_to_linear(v)), v);
        }
    }
}
