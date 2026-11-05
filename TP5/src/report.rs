//! Reporte de pixeles incorrectos por patron.

use std::path::Path;

use anyhow::{Context, Result};
use ndarray::Array2;

/// Consigna 1a.2: a lo sumo 1 pixel incorrecto por patron.
pub const TOLERANCE_PX: usize = 1;

/// Pixeles incorrectos por fila: se binariza en 0.5 tanto la reconstruccion como el objetivo.
pub fn wrong_pixel_counts(target: &Array2<f64>, recon: &Array2<f64>) -> Vec<usize> {
    target
        .rows()
        .into_iter()
        .zip(recon.rows())
        .map(|(t, r)| t.iter().zip(r.iter()).filter(|(&a, &b)| (a >= 0.5) != (b >= 0.5)).count())
        .collect()
}

#[derive(Debug, Clone)]
pub struct PatternResult {
    pub label: String,
    pub wrong_pixels: usize,
}

#[derive(Debug, Clone)]
pub struct Report {
    pub patterns: Vec<PatternResult>,
}

impl Report {
    pub fn new(labels: &[String], counts: &[usize]) -> Report {
        let patterns = labels
            .iter()
            .zip(counts)
            .map(|(l, &c)| PatternResult { label: l.clone(), wrong_pixels: c })
            .collect();
        Report { patterns }
    }

    pub fn max_wrong(&self) -> usize {
        self.patterns.iter().map(|p| p.wrong_pixels).max().unwrap_or(0)
    }

    pub fn n_within(&self, tol: usize) -> usize {
        self.patterns.iter().filter(|p| p.wrong_pixels <= tol).count()
    }

    pub fn render_table(&self) -> String {
        let mut out = String::from("idx  caracter  pixeles_incorrectos  estado");
        for (i, p) in self.patterns.iter().enumerate() {
            let estado = if p.wrong_pixels <= TOLERANCE_PX { "OK" } else { "FALLA" };
            out.push_str(&format!("\n{:>3}  {:<8}  {:>19}  {}", i, p.label, p.wrong_pixels, estado));
        }
        out
    }

    /// Linea final del Ej 1a: `RESULTADO: k/32 patrones ...`.
    pub fn result_line(&self) -> String {
        format!(
            "RESULTADO: {}/32 patrones con a lo sumo 1 pixel incorrecto (maximo {})",
            self.n_within(TOLERANCE_PX),
            self.max_wrong()
        )
    }

    pub fn summary_line(&self) -> String {
        format!(
            "resumen: {}/32 patrones con a lo sumo 1 pixel incorrecto, maximo {} pixeles incorrectos",
            self.n_within(TOLERANCE_PX),
            self.max_wrong()
        )
    }
}

/// Escribe `idx,label,wrong_pixels` con una fila por patron. Las etiquetas con coma o comillas
/// (por ejemplo `,` o `"`) se entrecomillan segun CSV.
pub fn write_report_csv(path: &Path, report: &Report) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("no se pudo crear {}", parent.display()))?;
    }
    let mut out = String::from("idx,label,wrong_pixels\n");
    for (i, p) in report.patterns.iter().enumerate() {
        let label = if p.label.contains(',') || p.label.contains('"') {
            format!("\"{}\"", p.label.replace('"', "\"\""))
        } else {
            p.label.clone()
        };
        out.push_str(&format!("{},{},{}\n", i, label, p.wrong_pixels));
    }
    std::fs::write(path, out).with_context(|| format!("no se pudo escribir {}", path.display()))
}
