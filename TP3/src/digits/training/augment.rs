//! On-the-fly data augmentation for square grayscale digit images.

use rand::Rng;

/// Writes a randomly translated and rotated copy of `source` into `target`.
///
/// Each output pixel is mapped back into the source image (inverse affine
/// transform around the image center) and sampled with bilinear
/// interpolation; pixels that fall outside the source are background (0).
pub(super) fn augment_into<R: Rng>(
    source: &[f64],
    target: &mut [f64],
    rng: &mut R,
    max_shift: f64,
    max_rotation_degrees: f64,
) {
    let side = (source.len() as f64).sqrt().round() as usize;
    debug_assert_eq!(side * side, source.len());
    let symmetric = |rng: &mut R, limit: f64| {
        if limit > 0.0 {
            rng.gen_range(-limit..=limit)
        } else {
            0.0
        }
    };
    let shift_x = symmetric(rng, max_shift);
    let shift_y = symmetric(rng, max_shift);
    let angle = symmetric(rng, max_rotation_degrees).to_radians();
    transform_into(source, target, side, shift_x, shift_y, angle);
}

fn transform_into(
    source: &[f64],
    target: &mut [f64],
    side: usize,
    shift_x: f64,
    shift_y: f64,
    angle: f64,
) {
    let center = (side as f64 - 1.0) / 2.0;
    let (sin, cos) = angle.sin_cos();
    for y in 0..side {
        for x in 0..side {
            let px = x as f64 - center - shift_x;
            let py = y as f64 - center - shift_y;
            let source_x = cos * px + sin * py + center;
            let source_y = -sin * px + cos * py + center;
            target[y * side + x] = bilinear(source, side, source_x, source_y);
        }
    }
}

fn bilinear(image: &[f64], side: usize, x: f64, y: f64) -> f64 {
    let x0 = x.floor();
    let y0 = y.floor();
    let fx = x - x0;
    let fy = y - y0;
    let pixel = |xi: f64, yi: f64| {
        if xi < 0.0 || yi < 0.0 || xi >= side as f64 || yi >= side as f64 {
            0.0
        } else {
            image[yi as usize * side + xi as usize]
        }
    };
    pixel(x0, y0) * (1.0 - fx) * (1.0 - fy)
        + pixel(x0 + 1.0, y0) * fx * (1.0 - fy)
        + pixel(x0, y0 + 1.0) * (1.0 - fx) * fy
        + pixel(x0 + 1.0, y0 + 1.0) * fx * fy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_transform_preserves_the_image() {
        let source = (0..16).map(|value| value as f64 / 15.0).collect::<Vec<_>>();
        let mut target = vec![0.0; 16];
        transform_into(&source, &mut target, 4, 0.0, 0.0, 0.0);
        assert_eq!(source, target);
    }

    #[test]
    fn integer_shift_moves_pixels_and_fills_background() {
        let mut source = vec![0.0; 9];
        source[4] = 1.0; // center of a 3x3 image
        let mut target = vec![0.0; 9];
        transform_into(&source, &mut target, 3, 1.0, 0.0, 0.0);
        assert_eq!(target[5], 1.0);
        assert_eq!(target.iter().sum::<f64>(), 1.0);
    }
}
