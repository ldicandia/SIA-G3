//! AE-04: subconjunto aprendido y evidencia de por que fallan los demas patrones.

use std::path::Path;

use anyhow::{bail, Context, Result};
use ndarray::Array2;

use crate::autoencoder::{Autoencoder, TrainHistory};
use crate::data::Dataset;
use crate::plots::plot_latent_scatter;
use crate::report::{Report, TOLERANCE_PX};

/// Nombres fijos de los archivos del espacio latente.
pub const LATENT_PNG: &str = "latent.png";
pub const LATENT_CSV: &str = "latent_codes.csv";

#[derive(Debug, Clone)]
pub struct FailedPattern {
    pub label: String,
    pub wrong_pixels: usize,
    pub nearest_label: String,
    pub latent_distance: f64,
    pub hamming_to_nearest: usize,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct SubsetAnalysis {
    pub learned: Vec<String>,
    pub failed: Vec<FailedPattern>,
    /// Par de patrones con los codigos latentes mas cercanos y su distancia.
    pub closest_pair: (String, String, f64),
    pub loss_plateau: bool,
    pub final_loss: f64,
}

fn latent_distance(codes: &Array2<f64>, i: usize, j: usize) -> f64 {
    codes
        .row(i)
        .iter()
        .zip(codes.row(j).iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
}

fn hamming(patterns: &Array2<f64>, i: usize, j: usize) -> usize {
    patterns
        .row(i)
        .iter()
        .zip(patterns.row(j).iter())
        .filter(|(&a, &b)| (a >= 0.5) != (b >= 0.5))
        .count()
}

/// Verdadero cuando la media del ultimo 10 % de la perdida mejoro menos de 1 % respecto de la
/// media de la ventana previa (el 10 % anterior).
fn detect_plateau(loss: &[f64]) -> bool {
    let w = loss.len() / 10;
    if w == 0 || loss.len() < 2 * w {
        return false;
    }
    let mean = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
    let last = mean(&loss[loss.len() - w..]);
    let prev = mean(&loss[loss.len() - 2 * w..loss.len() - w]);
    if prev <= 0.0 {
        return true;
    }
    (prev - last) / prev < 0.01
}

pub fn analyze(
    labels: &[String],
    patterns: &Array2<f64>,
    codes: &Array2<f64>,
    report: &Report,
    history: &TrainHistory,
) -> SubsetAnalysis {
    let n = labels.len();
    // Vecino mas cercano (en el espacio latente) de cada patron.
    let mut nearest = vec![(0usize, f64::INFINITY); n];
    let mut closest = (0usize, 0usize, f64::INFINITY);
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = latent_distance(codes, i, j);
            if d < nearest[i].1 {
                nearest[i] = (j, d);
            }
            if i < j && d < closest.2 {
                closest = (i, j, d);
            }
        }
    }
    let mut nn_dists: Vec<f64> = nearest.iter().map(|x| x.1).filter(|d| d.is_finite()).collect();
    nn_dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_nn = if nn_dists.is_empty() { 0.0 } else { nn_dists[nn_dists.len() / 2] };

    let loss_plateau = detect_plateau(&history.loss);
    let mut learned = Vec::new();
    let mut failed = Vec::new();
    for (i, p) in report.patterns.iter().enumerate() {
        if p.wrong_pixels <= TOLERANCE_PX {
            learned.push(p.label.clone());
            continue;
        }
        let (j, dist) = nearest[i];
        let reason = if dist < median_nn / 4.0 {
            format!(
                "los codigos latentes estan casi superpuestos con los de '{}' (distancia {:.4} < 1/4 de la mediana {:.4}), el decoder no puede separarlos",
                labels[j], dist, median_nn
            )
        } else if loss_plateau {
            format!(
                "el entrenamiento se estanco en una meseta de la perdida (mejora < 1 % en el ultimo 10 %, perdida final {:.6})",
                history.final_loss
            )
        } else {
            "con las epocas usadas no es concluyente: la perdida seguia bajando".to_string()
        };
        failed.push(FailedPattern {
            label: p.label.clone(),
            wrong_pixels: p.wrong_pixels,
            nearest_label: labels[j].clone(),
            latent_distance: dist,
            hamming_to_nearest: hamming(patterns, i, j),
            reason,
        });
    }
    let closest_pair = if n >= 2 {
        (labels[closest.0].clone(), labels[closest.1].clone(), closest.2)
    } else {
        (String::new(), String::new(), f64::NAN)
    };
    SubsetAnalysis { learned, failed, closest_pair, loss_plateau, final_loss: history.final_loss }
}

impl SubsetAnalysis {
    pub fn render(&self) -> String {
        if self.failed.is_empty() {
            return "SUBCONJUNTO: se aprendieron los 32 patrones (ninguno supera 1 pixel incorrecto); no queda ningun patron sin aprender"
                .to_string();
        }
        let mut out = format!("SUBCONJUNTO APRENDIDO: {}/32", self.learned.len());
        out.push_str(&format!("\naprendidos: {}", self.learned.join(" ")));
        out.push_str(&format!(
            "\nno aprendidos: {}",
            self.failed.iter().map(|f| f.label.as_str()).collect::<Vec<_>>().join(" ")
        ));
        for f in &self.failed {
            out.push_str(&format!(
                "\n  '{}': {} pixeles incorrectos | vecino latente '{}' a distancia {:.4} | Hamming {} | {}",
                f.label, f.wrong_pixels, f.nearest_label, f.latent_distance, f.hamming_to_nearest, f.reason
            ));
        }
        out.push_str(&format!(
            "\npar mas cercano en el latente: '{}' y '{}' a distancia {:.4}",
            self.closest_pair.0, self.closest_pair.1, self.closest_pair.2
        ));
        out.push_str(&format!(
            "\nmeseta de perdida: {} | perdida final {:.6}",
            if self.loss_plateau { "si" } else { "no" },
            self.final_loss
        ));
        out
    }
}

fn csv_field(label: &str) -> String {
    if label.contains(',') || label.contains('"') {
        format!("\"{}\"", label.replace('"', "\"\""))
    } else {
        label.to_string()
    }
}

/// Escribe `label,z1,z2` con una fila por patron (coordenadas a 6 decimales).
pub fn write_latent_csv(path: &Path, labels: &[String], codes: &Array2<f64>) -> Result<()> {
    if labels.len() != codes.nrows() || codes.ncols() != 2 {
        bail!("se esperaban {} codigos de 2 dimensiones y hay {}x{}", labels.len(), codes.nrows(), codes.ncols());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("no se pudo crear {}", parent.display()))?;
    }
    let mut out = String::from("label,z1,z2\n");
    for (l, row) in labels.iter().zip(codes.rows()) {
        out.push_str(&format!("{},{:.6},{:.6}\n", csv_field(l), row[0], row[1]));
    }
    std::fs::write(path, out).with_context(|| format!("no se pudo escribir {}", path.display()))
}

/// AE-05: codifica los 32 patrones, escribe `latent.png` y `latent_codes.csv` en `out_dir` y
/// devuelve los codigos.
pub fn export_latent(ae: &Autoencoder, ds: &Dataset, out_dir: &Path) -> Result<Array2<f64>> {
    let codes = ae.encode(&ds.x);
    let points: Vec<(String, f64, f64)> =
        ds.labels.iter().zip(codes.rows()).map(|(l, r)| (l.clone(), r[0], r[1])).collect();
    plot_latent_scatter(&out_dir.join(LATENT_PNG), &points)?;
    write_latent_csv(&out_dir.join(LATENT_CSV), &ds.labels, &codes)?;
    Ok(codes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic(failing: &[usize]) -> (Vec<String>, Array2<f64>, Array2<f64>, Report, TrainHistory) {
        let labels: Vec<String> = (0..32).map(|i| format!("p{i}")).collect();
        let patterns = Array2::from_shape_fn((32, 35), |(i, j)| ((i + j) % 2) as f64);
        // Codigos sobre una recta, salvo el patron 31 casi encima del 30.
        let codes = Array2::from_shape_fn((32, 2), |(i, k)| {
            if k == 1 {
                0.0
            } else if i == 31 {
                30.001
            } else {
                i as f64
            }
        });
        let counts: Vec<usize> = (0..32).map(|i| if failing.contains(&i) { 5 } else { 0 }).collect();
        let report = Report::new(&labels, &counts);
        let loss: Vec<f64> = (0..200).map(|e| 1.0 / (1.0 + e as f64)).collect();
        let history = TrainHistory {
            loss,
            evals: vec![],
            first_epoch_within: None,
            epochs_run: 200,
            final_loss: 0.005,
        };
        (labels, patterns, codes, report, history)
    }

    #[test]
    fn all_learned_renders_no_failures() {
        let (l, p, c, r, h) = synthetic(&[]);
        let a = analyze(&l, &p, &c, &r, &h);
        assert!(a.failed.is_empty());
        assert_eq!(a.learned.len(), 32);
        let text = a.render();
        assert!(text.starts_with("SUBCONJUNTO: se aprendieron los 32 patrones"));
        assert!(!text.contains("no aprendidos"));
    }

    #[test]
    fn two_failed_patterns_are_listed_with_reasons() {
        let (l, p, c, r, h) = synthetic(&[30, 31]);
        let a = analyze(&l, &p, &c, &r, &h);
        assert_eq!(a.failed.len(), 2);
        assert_eq!(a.learned.len(), 30);
        for f in &a.failed {
            assert!(f.reason.contains("superpuestos"), "{}", f.reason);
        }
        assert_eq!(a.failed[0].nearest_label, "p31");
        let text = a.render();
        assert!(text.contains("SUBCONJUNTO APRENDIDO: 30/32"));
        assert!(text.contains("no aprendidos: p30 p31"));
    }

    #[test]
    fn plateau_detection() {
        let flat: Vec<f64> = (0..200).map(|e| if e < 100 { 2.0 - e as f64 * 0.01 } else { 1.0 }).collect();
        assert!(detect_plateau(&flat));
        let decaying: Vec<f64> = (0..200).map(|e| (-0.02 * e as f64).exp()).collect();
        assert!(!detect_plateau(&decaying));
    }
}
