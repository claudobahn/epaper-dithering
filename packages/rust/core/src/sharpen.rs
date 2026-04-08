//! Pre-dithering sharpening to counteract error-diffusion blur.

/// Unsharp mask: sharpens by subtracting a box-blurred copy from the original.
///
/// `pixel = pixel + amount × (pixel − blurred)`
///
/// Operates on linear RGB pixels in place. A radius of 1 (3×3 kernel) with
/// amount 0.5–1.5 is recommended for e-paper error diffusion.
pub fn unsharp_mask(pixels: &mut [[f64; 3]], width: usize, height: usize, amount: f64, radius: usize) {
    if amount <= 0.0 || radius == 0 || pixels.is_empty() {
        return;
    }

    // Box blur
    let mut blurred = pixels.to_vec();
    let r = radius as i64;
    for y in 0..height {
        for x in 0..width {
            let mut sum = [0.0_f64; 3];
            let mut count = 0.0;
            for dy in -r..=r {
                for dx in -r..=r {
                    let nx = x as i64 + dx;
                    let ny = y as i64 + dy;
                    if nx >= 0 && nx < width as i64 && ny >= 0 && ny < height as i64 {
                        let idx = ny as usize * width + nx as usize;
                        sum[0] += pixels[idx][0];
                        sum[1] += pixels[idx][1];
                        sum[2] += pixels[idx][2];
                        count += 1.0;
                    }
                }
            }
            let idx = y * width + x;
            blurred[idx] = [sum[0] / count, sum[1] / count, sum[2] / count];
        }
    }

    // Sharpen: original + amount × (original − blurred)
    for (pixel, blur) in pixels.iter_mut().zip(blurred.iter()) {
        pixel[0] = (pixel[0] + amount * (pixel[0] - blur[0])).clamp(0.0, 1.0);
        pixel[1] = (pixel[1] + amount * (pixel[1] - blur[1])).clamp(0.0, 1.0);
        pixel[2] = (pixel[2] + amount * (pixel[2] - blur[2])).clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_image_unchanged() {
        let mut pixels = vec![[0.5, 0.5, 0.5]; 9];
        let original = pixels.clone();
        unsharp_mask(&mut pixels, 3, 3, 1.0, 1);
        for (p, o) in pixels.iter().zip(original.iter()) {
            let d: f64 = p.iter().zip(o.iter()).map(|(a, b)| (a - b).abs()).sum();
            assert!(d < 1e-10, "flat image should not be modified: d={d}");
        }
    }

    #[test]
    fn zero_amount_is_identity() {
        let mut pixels = vec![[0.2, 0.8, 0.4], [0.9, 0.1, 0.5], [0.3, 0.6, 0.7]];
        let original = pixels.clone();
        unsharp_mask(&mut pixels, 3, 1, 0.0, 1);
        assert_eq!(pixels, original);
    }

    #[test]
    fn sharpening_increases_edge_contrast() {
        // A sharp edge: left half dark, right half bright
        let mut pixels = vec![
            [0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.8, 0.8, 0.8], [0.8, 0.8, 0.8],
        ];
        let dark_before = pixels[1][0];
        let bright_before = pixels[2][0];
        unsharp_mask(&mut pixels, 4, 1, 1.0, 1);
        // The dark pixel next to the edge should get darker
        assert!(pixels[1][0] < dark_before, "dark side of edge should get darker");
        // The bright pixel next to the edge should get brighter
        assert!(pixels[2][0] > bright_before, "bright side of edge should get brighter");
    }
}
