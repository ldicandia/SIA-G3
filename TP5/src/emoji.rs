//! Dataset de emojis del Ej 2 (VAE-01), construido por codigo: 36 imagenes de 24 x 24 que combinan
//! 6 estilos de ojos con 6 estilos de boca dentro de un mismo contorno de cara. Modulo numerico:
//! no dibuja ni parsea argumentos.

use ndarray::{Array1, Array2, ArrayView1};

pub const EMOJI_SIDE: usize = 24;
pub const EMOJI_PIXELS: usize = EMOJI_SIDE * EMOJI_SIDE;
pub const EMOJI_COUNT: usize = 36;

/// Estilos de ojos (indice externo del dataset).
pub const EMOJI_EYES: [&str; 6] = ["dots", "happy", "wide", "wink", "closed", "x"];
/// Estilos de boca (indice interno del dataset).
pub const EMOJI_MOUTHS: [&str; 6] = ["smile", "grin", "flat", "frown", "open", "small"];

/// Submuestras por lado de cada pixel (supersampling 4 x 4).
const SUPER: usize = 4;

const EYE_Y: f64 = 9.8;
const EYE_LEFT_X: f64 = 8.3;
const EYE_RIGHT_X: f64 = 15.7;

#[derive(Debug, Clone)]
pub struct EmojiDataset {
    /// Nombres `{ojos}-{boca}`, con los ojos como bucle externo (indice = 6 * ojos + boca).
    pub names: Vec<String>,
    /// 36 x 576, fila mayor sobre (y, x); 1.0 = tinta completa, 0.0 = vacio.
    pub x: Array2<f64>,
}

/// Primitivas geometricas en unidades de pixel de salida, origen arriba a la izquierda, y hacia abajo.
#[derive(Debug, Clone, Copy)]
enum Shape {
    /// Sector de anillo: distancia al centro dentro de `thickness / 2` del radio y angulo en `[a0, a1]`
    /// (grados, atan2(dy, dx) modulo 360). Una vuelta completa es `a0 = 0`, `a1 = 360`.
    Ring { cx: f64, cy: f64, radius: f64, thickness: f64, a0: f64, a1: f64 },
    /// Capsula: puntos a distancia menor que `thickness / 2` del segmento.
    Capsule { x0: f64, y0: f64, x1: f64, y1: f64, thickness: f64 },
    Disc { cx: f64, cy: f64, radius: f64 },
    /// Semidisco inferior (y >= cy).
    LowerHalfDisc { cx: f64, cy: f64, radius: f64 },
    Ellipse { cx: f64, cy: f64, rx: f64, ry: f64 },
}

impl Shape {
    fn contains(&self, px: f64, py: f64) -> bool {
        match *self {
            Shape::Ring { cx, cy, radius, thickness, a0, a1 } => {
                let (dx, dy) = (px - cx, py - cy);
                let dist = (dx * dx + dy * dy).sqrt();
                if (dist - radius).abs() > thickness / 2.0 {
                    return false;
                }
                if a1 - a0 >= 360.0 {
                    return true;
                }
                let mut angle = dy.atan2(dx).to_degrees();
                if angle < 0.0 {
                    angle += 360.0;
                }
                angle >= a0 && angle <= a1
            }
            Shape::Capsule { x0, y0, x1, y1, thickness } => {
                let (vx, vy) = (x1 - x0, y1 - y0);
                let len2 = vx * vx + vy * vy;
                let t = if len2 == 0.0 { 0.0 } else { (((px - x0) * vx + (py - y0) * vy) / len2).clamp(0.0, 1.0) };
                let (qx, qy) = (x0 + t * vx, y0 + t * vy);
                let (dx, dy) = (px - qx, py - qy);
                (dx * dx + dy * dy).sqrt() <= thickness / 2.0
            }
            Shape::Disc { cx, cy, radius } => {
                let (dx, dy) = (px - cx, py - cy);
                dx * dx + dy * dy <= radius * radius
            }
            Shape::LowerHalfDisc { cx, cy, radius } => {
                let (dx, dy) = (px - cx, py - cy);
                py >= cy && dx * dx + dy * dy <= radius * radius
            }
            Shape::Ellipse { cx, cy, rx, ry } => {
                let (dx, dy) = ((px - cx) / rx, (py - cy) / ry);
                dx * dx + dy * dy <= 1.0
            }
        }
    }
}

fn face_outline() -> Vec<Shape> {
    vec![Shape::Ring { cx: 12.0, cy: 12.0, radius: 10.6, thickness: 1.4, a0: 0.0, a1: 360.0 }]
}

fn dot_eye(x: f64) -> Vec<Shape> {
    vec![Shape::Disc { cx: x, cy: EYE_Y, radius: 1.4 }]
}

fn happy_eye(x: f64) -> Vec<Shape> {
    vec![Shape::Ring { cx: x, cy: EYE_Y + 1.2, radius: 2.2, thickness: 1.1, a0: 205.0, a1: 335.0 }]
}

fn eye_shapes(style: usize, x: f64, is_left: bool) -> Vec<Shape> {
    match style {
        0 => dot_eye(x),
        1 => happy_eye(x),
        2 => vec![
            Shape::Ring { cx: x, cy: EYE_Y, radius: 2.2, thickness: 1.0, a0: 0.0, a1: 360.0 },
            Shape::Disc { cx: x, cy: EYE_Y, radius: 0.9 },
        ],
        3 => {
            if is_left {
                dot_eye(x)
            } else {
                happy_eye(x)
            }
        }
        4 => vec![Shape::Capsule { x0: x - 1.8, y0: EYE_Y, x1: x + 1.8, y1: EYE_Y, thickness: 1.1 }],
        5 => vec![
            Shape::Capsule { x0: x - 1.6, y0: EYE_Y - 1.6, x1: x + 1.6, y1: EYE_Y + 1.6, thickness: 1.0 },
            Shape::Capsule { x0: x - 1.6, y0: EYE_Y + 1.6, x1: x + 1.6, y1: EYE_Y - 1.6, thickness: 1.0 },
        ],
        _ => unreachable!("estilo de ojos fuera de rango"),
    }
}

fn mouth_shapes(style: usize) -> Vec<Shape> {
    match style {
        0 => vec![Shape::Ring { cx: 12.0, cy: 13.0, radius: 5.5, thickness: 1.2, a0: 25.0, a1: 155.0 }],
        1 => vec![
            Shape::LowerHalfDisc { cx: 12.0, cy: 14.0, radius: 5.6 },
            Shape::Capsule { x0: 6.4, y0: 14.0, x1: 17.6, y1: 14.0, thickness: 1.0 },
        ],
        2 => vec![Shape::Capsule { x0: 8.5, y0: 17.0, x1: 15.5, y1: 17.0, thickness: 1.2 }],
        3 => vec![Shape::Ring { cx: 12.0, cy: 21.5, radius: 5.0, thickness: 1.2, a0: 215.0, a1: 325.0 }],
        4 => vec![Shape::Ellipse { cx: 12.0, cy: 17.5, rx: 2.4, ry: 3.0 }],
        5 => vec![Shape::Ring { cx: 12.0, cy: 15.0, radius: 3.0, thickness: 1.2, a0: 25.0, a1: 155.0 }],
        _ => unreachable!("estilo de boca fuera de rango"),
    }
}

/// Renderiza las primitivas con supersampling: cada pixel es la media de una grilla de 4 x 4
/// submuestras tomadas en los centros de las subceldas.
fn render(shapes: &[Shape]) -> Vec<f64> {
    let mut out = Vec::with_capacity(EMOJI_PIXELS);
    for y in 0..EMOJI_SIDE {
        for x in 0..EMOJI_SIDE {
            let mut ink = 0usize;
            for sy in 0..SUPER {
                for sx in 0..SUPER {
                    let px = x as f64 + (sx as f64 + 0.5) / SUPER as f64;
                    let py = y as f64 + (sy as f64 + 0.5) / SUPER as f64;
                    if shapes.iter().any(|s| s.contains(px, py)) {
                        ink += 1;
                    }
                }
            }
            out.push(ink as f64 / (SUPER * SUPER) as f64);
        }
    }
    out
}

/// Construye los 36 emojis. Aritmetica pura sin azar: dos construcciones son identicas bit a bit.
pub fn build_emojis() -> EmojiDataset {
    let mut names = Vec::with_capacity(EMOJI_COUNT);
    let mut data = Vec::with_capacity(EMOJI_COUNT * EMOJI_PIXELS);
    for (e, eyes) in EMOJI_EYES.iter().enumerate() {
        for (m, mouth) in EMOJI_MOUTHS.iter().enumerate() {
            let mut shapes = face_outline();
            shapes.extend(eye_shapes(e, EYE_LEFT_X, true));
            shapes.extend(eye_shapes(e, EYE_RIGHT_X, false));
            shapes.extend(mouth_shapes(m));
            names.push(format!("{eyes}-{mouth}"));
            data.extend(render(&shapes));
        }
    }
    let x = Array2::from_shape_vec((EMOJI_COUNT, EMOJI_PIXELS), data).expect("forma 36 x 576");
    EmojiDataset { names, x }
}

/// Posiciones donde los dos vectores difieren tras binarizar en 0.5 (misma regla que
/// `report::wrong_pixel_counts`).
pub fn hamming(a: ArrayView1<f64>, b: ArrayView1<f64>) -> usize {
    a.iter().zip(b.iter()).filter(|(&p, &q)| (p >= 0.5) != (q >= 0.5)).count()
}

/// Indice del emoji del dataset a menor distancia Hamming (el primero ante empates) y esa distancia.
pub fn nearest_emoji(ds: &EmojiDataset, pattern: ArrayView1<f64>) -> (usize, usize) {
    let mut best = (0usize, usize::MAX);
    for (i, row) in ds.x.rows().into_iter().enumerate() {
        let d = hamming(row, pattern);
        if d < best.1 {
            best = (i, d);
        }
    }
    best
}

/// Separacion del dataset: distancia de cada emoji a su vecino mas cercano.
#[derive(Debug, Clone, PartialEq)]
pub struct Separation {
    pub min: usize,
    pub median: f64,
    /// Mediana redondeada hacia arriba: radio con el que un patron se considera "cerca" de un emoji.
    pub near_radius: usize,
}

pub fn separation(ds: &EmojiDataset) -> Separation {
    let n = ds.x.nrows();
    let mut nearest: Vec<usize> = (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| j != i)
                .map(|j| hamming(ds.x.row(i), ds.x.row(j)))
                .min()
                .expect("hay mas de un emoji")
        })
        .collect();
    nearest.sort_unstable();
    let median = if n % 2 == 1 {
        nearest[n / 2] as f64
    } else {
        (nearest[n / 2 - 1] + nearest[n / 2]) as f64 / 2.0
    };
    Separation { min: nearest[0], median, near_radius: median.ceil() as usize }
}

/// 24 lineas de 24 caracteres: `#` si el valor es al menos 0.5, `.` si no.
pub fn format_emoji(pattern: ArrayView1<f64>) -> String {
    (0..EMOJI_SIDE)
        .map(|r| {
            (0..EMOJI_SIDE)
                .map(|c| if pattern[r * EMOJI_SIDE + c] >= 0.5 { '#' } else { '.' })
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Une las lineas de varias representaciones con dos espacios entre ellas.
pub fn format_emojis_side_by_side(patterns: &[Array1<f64>]) -> String {
    let rendered: Vec<String> = patterns.iter().map(|p| format_emoji(p.view())).collect();
    (0..EMOJI_SIDE)
        .map(|line| rendered.iter().map(|r| r.lines().nth(line).unwrap_or("")).collect::<Vec<_>>().join("  "))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn has_36_unique_names_in_eyes_major_order() {
        let ds = build_emojis();
        assert_eq!(ds.x.dim(), (36, 576));
        assert_eq!(ds.names.len(), 36);
        let unique: HashSet<&String> = ds.names.iter().collect();
        assert_eq!(unique.len(), 36);
        assert_eq!(ds.names[0], "dots-smile");
        assert_eq!(ds.names[6], "happy-smile");
        assert_eq!(ds.names[35], "x-small");
    }

    #[test]
    fn pixels_are_in_unit_interval_and_every_image_has_ink() {
        let ds = build_emojis();
        assert!(ds.x.iter().all(|&v| (0.0..=1.0).contains(&v)));
        for row in ds.x.rows() {
            assert!(row.iter().any(|&v| v > 0.0));
        }
    }

    #[test]
    fn build_is_deterministic_bit_for_bit() {
        let a = build_emojis();
        let b = build_emojis();
        assert_eq!(a.names, b.names);
        assert!(a.x.iter().zip(b.x.iter()).all(|(p, q)| p.to_bits() == q.to_bits()));
    }

    #[test]
    fn emojis_are_pairwise_distinct_after_binarizing() {
        let ds = build_emojis();
        for i in 0..36 {
            for j in (i + 1)..36 {
                let d = hamming(ds.x.row(i), ds.x.row(j));
                assert!(d >= 5, "{} y {} difieren en solo {d} pixeles", ds.names[i], ds.names[j]);
            }
        }
    }

    #[test]
    fn rows_above_the_features_are_identical_across_emojis() {
        let ds = build_emojis();
        let top = 6 * EMOJI_SIDE;
        for i in 1..36 {
            assert_eq!(ds.x.row(i).slice(ndarray::s![..top]), ds.x.row(0).slice(ndarray::s![..top]));
        }
    }

    #[test]
    fn nearest_emoji_of_a_dataset_row_is_itself_at_distance_zero() {
        let ds = build_emojis();
        for i in 0..36 {
            assert_eq!(nearest_emoji(&ds, ds.x.row(i)), (i, 0));
        }
    }

    #[test]
    fn separation_matches_an_independent_computation() {
        let ds = build_emojis();
        let mut per_emoji = Vec::new();
        for i in 0..36 {
            let mut best = usize::MAX;
            for j in 0..36 {
                if i == j {
                    continue;
                }
                let mut d = 0;
                for k in 0..576 {
                    if (ds.x[[i, k]] >= 0.5) != (ds.x[[j, k]] >= 0.5) {
                        d += 1;
                    }
                }
                best = best.min(d);
            }
            per_emoji.push(best);
        }
        per_emoji.sort();
        let median = (per_emoji[17] + per_emoji[18]) as f64 / 2.0;
        let s = separation(&ds);
        assert_eq!(s.min, per_emoji[0]);
        assert_eq!(s.median, median);
        assert_eq!(s.near_radius, median.ceil() as usize);
    }

    #[test]
    fn ascii_rendering_has_24_lines_of_24_characters() {
        let ds = build_emojis();
        let text = format_emoji(ds.x.row(0));
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 24);
        assert!(lines.iter().all(|l| l.chars().count() == 24));
        let rows = vec![ds.x.row(0).to_owned(), ds.x.row(1).to_owned()];
        let side = format_emojis_side_by_side(&rows);
        assert_eq!(side.lines().count(), 24);
        assert!(side.lines().all(|l| l.chars().count() == 50));
    }
}
