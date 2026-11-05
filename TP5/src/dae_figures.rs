//! Exportadores de las figuras del denoising (Ej 1b): ejemplos de letras distorsionadas por nivel.
//! Paralelo a `analysis` y `generate`; el dibujo lo hace `plots::plot_pattern_rows`.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Result};
use ndarray::{Array1, Array2};

use crate::autoencoder::Autoencoder;
use crate::data::{format_grid, Dataset, N_PIXELS};
use crate::denoising::LevelStats;
use crate::noise::{distort, level_rng, validate_level};
use crate::plots::{plot_pattern_rows, plot_series, PatternRow, Series};
use crate::report::wrong_pixel_counts;

/// Letras que se muestran como ejemplo, en este orden.
pub const EXAMPLE_LABELS: [&str; 8] = ["a", "b", "e", "g", "k", "s", "w", "z"];

/// Semilla por defecto del ruido de los ejemplos.
pub const EXAMPLES_NOISE_SEED: u64 = 0xe8a3_0003;

/// Nombre fijo del archivo de ejemplos (no se interpola texto del usuario en la ruta).
pub const NOISE_EXAMPLES_PNG: &str = "noise_examples.png";

/// Nombres fijos de los resultados por nivel.
pub const LEVELS_CSV: &str = "dae_levels.csv";
pub const ERROR_CHART_PNG: &str = "dae_error_vs_noise.png";
pub const WITHIN_CHART_PNG: &str = "dae_within_vs_noise.png";

/// Encabezado del CSV por nivel.
pub const LEVELS_CSV_HEADER: &str =
    "level,n_samples,input_mean_wrong,output_mean_wrong,output_max_wrong,input_frac_within_1px,frac_within_1px";

/// Escribe una fila por nivel con las estadisticas medidas sobre distorsiones no vistas.
pub fn write_levels_csv(path: &Path, stats: &[LevelStats]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| anyhow!("no se pudo crear {}: {e}", parent.display()))?;
    }
    let mut out = String::from(LEVELS_CSV_HEADER);
    out.push('\n');
    for s in stats {
        out.push_str(&format!(
            "{:.2},{},{:.6},{:.6},{},{:.6},{:.6}\n",
            s.level, s.n_samples, s.input_mean_wrong, s.output_mean_wrong, s.output_max_wrong, s.input_frac_within,
            s.frac_within
        ));
    }
    std::fs::write(path, out).map_err(|e| anyhow!("no se pudo escribir {}: {e}", path.display()))?;
    Ok(())
}

/// Dos graficos contra el nivel de ruido: pixeles incorrectos medios y porcentaje de muestras con
/// a lo sumo 1 pixel incorrecto, de la entrada ruidosa y de la salida del autoencoder.
pub fn export_level_charts(stats: &[LevelStats], out_dir: &Path) -> Result<Vec<PathBuf>> {
    if stats.len() < 2 {
        bail!("se necesitan al menos 2 niveles para graficar una linea (hay {})", stats.len());
    }
    let x_label = "nivel de ruido (probabilidad de invertir cada pixel)";
    let series = |input: Vec<(f64, f64)>, output: Vec<(f64, f64)>| {
        vec![
            Series { name: "entrada ruidosa (sin autoencoder)".to_string(), points: input },
            Series { name: "salida del autoencoder".to_string(), points: output },
        ]
    };
    let error_path = out_dir.join(ERROR_CHART_PNG);
    plot_series(
        &error_path,
        "Ej 1b - pixeles incorrectos por nivel de ruido (distorsiones no vistas)",
        x_label,
        "pixeles incorrectos por patron (media)",
        &series(
            stats.iter().map(|s| (s.level, s.input_mean_wrong)).collect(),
            stats.iter().map(|s| (s.level, s.output_mean_wrong)).collect(),
        ),
        false,
    )?;
    let within_path = out_dir.join(WITHIN_CHART_PNG);
    plot_series(
        &within_path,
        "Ej 1b - muestras con a lo sumo 1 pixel incorrecto",
        x_label,
        "porcentaje de muestras",
        &series(
            stats.iter().map(|s| (s.level, s.input_frac_within * 100.0)).collect(),
            stats.iter().map(|s| (s.level, s.frac_within * 100.0)).collect(),
        ),
        false,
    )?;
    Ok(vec![error_path, within_path])
}

/// Nombre fijo `dae_examples_p{nivel}.png`; el nivel ya paso `validate_level`, asi que solo hay
/// digitos y un punto.
pub fn dae_examples_filename(level: f64) -> String {
    format!("dae_examples_p{level:.2}.png")
}

/// Tres filas sobre las mismas 8 letras: original, entrada ruidosa y salida reconstruida, con los
/// pixeles incorrectos contra el patron limpio como pie.
pub fn build_dae_example_rows(ae: &Autoencoder, ex: &NoiseExamples, level_index: usize) -> Vec<PatternRow> {
    let level = ex.levels[level_index];
    let noisy = &ex.noisy[level_index];
    let recon = ae.reconstruct(noisy);
    let bin = recon.mapv(|v| if v >= 0.5 { 1.0 } else { 0.0 });
    let rows_of = |m: &Array2<f64>| m.rows().into_iter().map(|r| r.to_owned()).collect::<Vec<_>>();
    let caps = |counts: Vec<usize>| counts.into_iter().map(|k| format!("{k} px")).collect::<Vec<_>>();
    vec![
        PatternRow {
            label: "original".to_string(),
            cells: ex.labels.iter().cloned().zip(rows_of(&ex.clean)).collect(),
        },
        PatternRow {
            label: format!("entrada ruidosa p={level:.2}"),
            cells: caps(wrong_pixel_counts(&ex.clean, noisy)).into_iter().zip(rows_of(noisy)).collect(),
        },
        PatternRow {
            label: "salida reconstruida".to_string(),
            cells: caps(wrong_pixel_counts(&ex.clean, &bin)).into_iter().zip(rows_of(&bin)).collect(),
        },
    ]
}

/// Escribe una imagen por nivel de `ex`, en el orden de los niveles.
pub fn export_dae_examples(ae: &Autoencoder, ex: &NoiseExamples, out_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::with_capacity(ex.levels.len());
    for (li, &level) in ex.levels.iter().enumerate() {
        let path = out_dir.join(dae_examples_filename(level));
        plot_pattern_rows(
            &path,
            &format!("Ej 1b - nivel de ruido p={level:.2}: original, entrada ruidosa y salida del autoencoder"),
            &build_dae_example_rows(ae, ex, li),
        )?;
        paths.push(path);
    }
    Ok(paths)
}

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
