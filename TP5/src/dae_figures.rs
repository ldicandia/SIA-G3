//! Exportadores de las figuras del denoising (Ej 1b): ejemplos de letras distorsionadas por nivel.
//! Paralelo a `analysis` y `generate`; el dibujo lo hace `plots::plot_pattern_rows`.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Result};
use ndarray::{Array1, Array2};

use crate::data::{format_grid, Dataset, N_PIXELS};
use crate::noise::{distort, level_rng, validate_level};
use crate::plots::{plot_pattern_rows, PatternRow};

/// Letras que se muestran como ejemplo, en este orden.
pub const EXAMPLE_LABELS: [&str; 8] = ["a", "b", "e", "g", "k", "s", "w", "z"];

/// Semilla por defecto del ruido de los ejemplos.
pub const EXAMPLES_NOISE_SEED: u64 = 0xe8a3_0003;

/// Nombre fijo del archivo de ejemplos (no se interpola texto del usuario en la ruta).
pub const NOISE_EXAMPLES_PNG: &str = "noise_examples.png";

/// Letras limpias y su version distorsionada a cada nivel.
#[derive(Debug, Clone)]
pub struct NoiseExamples {
    pub labels: Vec<String>,
    /// 8 x 35: las filas del dataset cuya etiqueta esta en `EXAMPLE_LABELS`, en ese orden.
    pub clean: Array2<f64>,
    pub levels: Vec<f64>,
    /// Una matriz de 8 x 35 por nivel, en el orden de `levels`.
    pub noisy: Vec<Array2<f64>>,
}

impl NoiseExamples {
    /// Pixeles que difieren de la letra limpia, por letra, para el nivel de indice `level_index`.
    pub fn inverted_pixels(&self, level_index: usize) -> Vec<usize> {
        self.noisy[level_index]
            .rows()
            .into_iter()
            .zip(self.clean.rows())
            .map(|(n, c)| n.iter().zip(c.iter()).filter(|(&a, &b)| (a >= 0.5) != (b >= 0.5)).count())
            .collect()
    }
}

/// Arma los ejemplos: cada nivel distorsiona `clean` con `level_rng(seed, nivel)`.
pub fn build_noise_examples(ds: &Dataset, levels: &[f64], seed: u64) -> Result<NoiseExamples> {
    for &l in levels {
        validate_level(l)?;
    }
    let mut clean = Array2::zeros((EXAMPLE_LABELS.len(), N_PIXELS));
    for (k, label) in EXAMPLE_LABELS.iter().enumerate() {
        let i = ds
            .labels
            .iter()
            .position(|l| l == label)
            .ok_or_else(|| anyhow!("la letra de ejemplo '{label}' no esta en el dataset"))?;
        clean.row_mut(k).assign(&ds.x.row(i));
    }
    let noisy = levels.iter().map(|&l| distort(&clean, l, &mut level_rng(seed, l))).collect();
    Ok(NoiseExamples {
        labels: EXAMPLE_LABELS.iter().map(|s| s.to_string()).collect(),
        clean,
        levels: levels.to_vec(),
        noisy,
    })
}

/// Dibuja cada patron como 7 lineas y une las lineas de todos los patrones con dos espacios.
pub fn format_grids_side_by_side(patterns: &[Array1<f64>]) -> String {
    let grids: Vec<String> = patterns.iter().map(|p| format_grid(p.view())).collect();
    (0..crate::data::ROWS)
        .map(|r| grids.iter().map(|g| g.lines().nth(r).unwrap_or("")).collect::<Vec<_>>().join("  "))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Escribe `noise_examples.png` en `out_dir`: una fila `original` y una fila `p=..` por nivel,
/// con la cantidad de pixeles invertidos como pie de cada letra.
pub fn export_noise_examples(ex: &NoiseExamples, out_dir: &Path) -> Result<PathBuf> {
    if ex.levels.len() != ex.noisy.len() {
        bail!("hay {} niveles y {} matrices ruidosas", ex.levels.len(), ex.noisy.len());
    }
    let mut rows = vec![PatternRow {
        label: "original".to_string(),
        cells: ex.labels.iter().cloned().zip(ex.clean.rows().into_iter().map(|r| r.to_owned())).collect(),
    }];
    for (li, &level) in ex.levels.iter().enumerate() {
        let counts = ex.inverted_pixels(li);
        rows.push(PatternRow {
            label: format!("p={level:.2}"),
            cells: ex.noisy[li]
                .rows()
                .into_iter()
                .zip(counts)
                .map(|(r, k)| (format!("{k} px"), r.to_owned()))
                .collect(),
        });
    }
    let path = out_dir.join(NOISE_EXAMPLES_PNG);
    plot_pattern_rows(&path, "Ej 1b - ejemplos de distorsion por nivel de ruido", &rows)?;
    Ok(path)
}
