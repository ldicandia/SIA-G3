//! AE-06: generar una letra nueva decodificando un punto del plano latente que no es el codigo
//! de ningun patron de entrenamiento. Este modulo no dibuja la grilla por su cuenta: delega en
//! `plots::plot_pattern_grid`.

use std::collections::HashSet;
use std::path::Path;

use anyhow::{bail, Context, Result};
use ndarray::{Array1, Array2};

use crate::autoencoder::Autoencoder;
use crate::data::Dataset;
use crate::plots::plot_pattern_grid;
use crate::report::TOLERANCE_PX;

/// Una letra decodificada cuenta como nueva solo si difiere de todos los patrones de
/// entrenamiento en mas de la tolerancia de 1 pixel (no se hace pasar por una reconstruccion
/// ligeramente defectuosa).
pub const NOVELTY_MIN_HAMMING: usize = TOLERANCE_PX + 1;

/// Magnitud maxima aceptada para una coordenada latente pedida por el usuario (T-01-06).
pub const MAX_POINT_ABS: f64 = 1e6;

/// Nombre fijo del archivo de la letra nueva.
pub const NEW_LETTER_PNG: &str = "new_letter.png";

#[derive(Debug, Clone)]
pub struct NovelLetter {
    pub z: [f64; 2],
    /// Salida del decoder binarizada (1.0 cuando la salida es al menos 0.5).
    pub pattern: Array1<f64>,
    /// Menor distancia de Hamming (en pixeles) a los 32 patrones de entrenamiento.
    pub min_hamming: usize,
    pub nearest_label: String,
    pub nearest_index: usize,
    /// Distancia euclidea de `z` al codigo latente de entrenamiento mas cercano.
    pub latent_distance: f64,
}

fn binarize(v: &Array1<f64>) -> Array1<f64> {
    v.mapv(|x| if x >= 0.5 { 1.0 } else { 0.0 })
}

fn hamming_to_row(pattern: &Array1<f64>, ds: &Dataset, row: usize) -> usize {
    pattern.iter().zip(ds.x.row(row).iter()).filter(|(&a, &b)| (a >= 0.5) != (b >= 0.5)).count()
}

/// Decodifica un punto del plano latente y binariza la salida.
pub fn decode_point(ae: &Autoencoder, z: [f64; 2]) -> Array1<f64> {
    let input = Array2::from_shape_vec((1, 2), vec![z[0], z[1]]).expect("forma 1x2");
    binarize(&ae.decode(&input).row(0).to_owned())
}

fn build_letter(ds: &Dataset, codes: &Array2<f64>, z: [f64; 2], pattern: Array1<f64>) -> NovelLetter {
    let mut best = (0usize, usize::MAX);
    for i in 0..ds.x.nrows() {
        let h = hamming_to_row(&pattern, ds, i);
        if h < best.1 {
            best = (i, h);
        }
    }
    let latent_distance = codes
        .rows()
        .into_iter()
        .map(|r| ((r[0] - z[0]).powi(2) + (r[1] - z[1]).powi(2)).sqrt())
        .fold(f64::INFINITY, f64::min);
    NovelLetter {
        z,
        pattern,
        min_hamming: best.1,
        nearest_label: ds.labels[best.0].clone(),
        nearest_index: best.0,
        latent_distance,
    }
}

/// Completa todos los campos para un punto cualquiera (`min_hamming` 0 significa que la
/// decodificacion coincide con un patron de entrenamiento).
pub fn classify_point(ae: &Autoencoder, ds: &Dataset, codes: &Array2<f64>, z: [f64; 2]) -> NovelLetter {
    build_letter(ds, codes, z, decode_point(ae, z))
}

/// Recorre una grilla `steps` x `steps` sobre la caja de los codigos de entrenamiento (expandida
/// `pad_frac` de su rango a cada lado), decodifica todo en un solo lote y devuelve, en orden de
/// barrido, la primera aparicion de cada patron distinto que no sea uno de los 32.
pub fn scan_novel(ae: &Autoencoder, ds: &Dataset, steps: usize, pad_frac: f64) -> Vec<NovelLetter> {
    let codes = ae.encode(&ds.x);
    let axis = |k: usize| {
        let lo = codes.column(k).iter().copied().fold(f64::INFINITY, f64::min);
        let hi = codes.column(k).iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let pad = (hi - lo) * pad_frac;
        (lo - pad, hi + pad)
    };
    let (z1_lo, z1_hi) = axis(0);
    let (z2_lo, z2_hi) = axis(1);
    let steps = steps.max(1);
    let at = |lo: f64, hi: f64, i: usize| if steps == 1 { (lo + hi) / 2.0 } else { lo + (hi - lo) * i as f64 / (steps - 1) as f64 };

    let mut points = Vec::with_capacity(steps * steps);
    for i in 0..steps {
        for j in 0..steps {
            points.push([at(z1_lo, z1_hi, i), at(z2_lo, z2_hi, j)]);
        }
    }
    let grid = Array2::from_shape_fn((points.len(), 2), |(r, c)| points[r][c]);
    let decoded = ae.decode(&grid);

    let training: HashSet<Vec<bool>> =
        ds.x.rows().into_iter().map(|r| r.iter().map(|&v| v >= 0.5).collect()).collect();
    let mut seen: HashSet<Vec<bool>> = HashSet::new();
    let mut out = Vec::new();
    for (r, z) in decoded.rows().into_iter().zip(&points) {
        let bits: Vec<bool> = r.iter().map(|&v| v >= 0.5).collect();
        if training.contains(&bits) || !seen.insert(bits) {
            continue;
        }
        out.push(build_letter(ds, &codes, *z, binarize(&r.to_owned())));
    }
    out
}

/// Candidatos validos (a al menos `NOVELTY_MIN_HAMMING` de todo patron de entrenamiento) en orden
/// de ranking: menor `min_hamming` (lo mas parecido a una letra real sin serlo), luego menor
/// `latent_distance`, luego orden de barrido.
fn ranked(cands: &[NovelLetter]) -> Vec<&NovelLetter> {
    let mut v: Vec<&NovelLetter> = cands.iter().filter(|c| c.min_hamming >= NOVELTY_MIN_HAMMING).collect();
    // sort_by es estable: el orden de barrido desempata.
    v.sort_by(|a, b| {
        a.min_hamming
            .cmp(&b.min_hamming)
            .then(a.latent_distance.partial_cmp(&b.latent_distance).unwrap_or(std::cmp::Ordering::Equal))
    });
    v
}

pub fn pick_new_letter(cands: &[NovelLetter]) -> Option<&NovelLetter> {
    ranked(cands).into_iter().next()
}

pub fn top_candidates(cands: &[NovelLetter], n: usize) -> Vec<&NovelLetter> {
    ranked(cands).into_iter().take(n).collect()
}

/// Genera la letra nueva y escribe `new_letter.png` en `out_dir`. Sin `point` barre el plano
/// (relleno 0.1) y elige; con `point` valida y clasifica ese punto.
pub fn export_new_letter(
    ae: &Autoencoder,
    ds: &Dataset,
    out_dir: &Path,
    point: Option<[f64; 2]>,
    steps: usize,
) -> Result<NovelLetter> {
    let letter = match point {
        None => {
            let cands = scan_novel(ae, ds, steps, 0.1);
            match pick_new_letter(&cands) {
                Some(l) => l.clone(),
                None => bail!(
                    "no se encontro ninguna letra a al menos {NOVELTY_MIN_HAMMING} pixeles de los 32 patrones de entrenamiento ({} patrones distintos explorados con una grilla de {steps}x{steps})",
                    cands.len()
                ),
            }
        }
        Some(z) => {
            if z.iter().any(|v| !v.is_finite()) {
                bail!("el punto latente debe tener coordenadas finitas (se recibio ({}, {}))", z[0], z[1]);
            }
            if z.iter().any(|v| v.abs() > MAX_POINT_ABS) {
                bail!("las coordenadas del punto latente deben tener magnitud de a lo sumo {MAX_POINT_ABS:e} (se recibio ({}, {}))", z[0], z[1]);
            }
            classify_point(ae, ds, &ae.encode(&ds.x), z)
        }
    };
    let items = vec![
        ("nueva letra".to_string(), letter.pattern.clone()),
        (format!("mas cercana: {}", letter.nearest_label), ds.x.row(letter.nearest_index).to_owned()),
    ];
    plot_pattern_grid(&out_dir.join(NEW_LETTER_PNG), &items).context("no se pudo escribir new_letter.png")?;
    Ok(letter)
}
