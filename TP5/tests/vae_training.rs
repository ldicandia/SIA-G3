//! Entrenamiento del VAE: comportamiento de la perdida, determinismo, tabla, CSV, grafico,
//! validacion de epocas y resumen de reconstruccion.

use std::path::PathBuf;
use std::sync::OnceLock;

use tp5::ores::{load_ores, OreDataset};
use tp5::vae::{
    chosen_vae_config, loss_summary_line, reconstruction_line, reconstruction_summary, render_loss_table, train_vae,
    validate_epochs, Vae, VaeConfig, VaeHistory,
};
use tp5::vae_figures::{export_loss_chart, write_loss_csv, LOSS_PNG};

const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

fn unique_tmp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tp5-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn config_with(epochs: usize) -> VaeConfig {
    let mut cfg = chosen_vae_config();
    cfg.epochs = epochs;
    cfg
}

const QUICK_EPOCHS: usize = 300;

fn quick() -> &'static (OreDataset, Vae, VaeHistory) {
    static MODEL: OnceLock<(OreDataset, Vae, VaeHistory)> = OnceLock::new();
    MODEL.get_or_init(|| {
        let ds = ores();
        let (vae, hist) = train_vae(&ds, &config_with(QUICK_EPOCHS)).unwrap();
        (ds, vae, hist)
    })
}

fn same_bits(a: &ndarray::Array2<f64>, b: &ndarray::Array2<f64>) -> bool {
    a.dim() == b.dim() && a.iter().zip(b.iter()).all(|(p, q)| p.to_bits() == q.to_bits())
}

fn weights_equal(a: &Vae, b: &Vae) -> bool {
    let layers = |v: &Vae| v.encoder.layers.iter().chain(v.decoder.layers.iter()).cloned().collect::<Vec<_>>();
    let (la, lb) = (layers(a), layers(b));
    la.len() == lb.len()
        && la.iter().zip(&lb).all(|(x, y)| {
            same_bits(&x.w, &y.w) && x.b.iter().zip(y.b.iter()).all(|(p, q)| p.to_bits() == q.to_bits())
        })
}

#[test]
fn loss_decreases_and_kl_stays_positive() {
    let (_, _, hist) = quick();
    assert_eq!(hist.recon.len(), QUICK_EPOCHS);
    let last = QUICK_EPOCHS - 1;
    // Con objetivos en gris la BCE no baja de la entropia de los objetivos: se exige recorrer al menos la
    // mitad de la distancia entre la perdida inicial y ese piso.
    let ds = &quick().0;
    let h = |p: f64| if p <= 0.0 || p >= 1.0 { 0.0 } else { -(p * p.ln() + (1.0 - p) * (1.0 - p).ln()) };
    let floor = ds.x.iter().map(|&p| h(p)).sum::<f64>() / ds.x.nrows() as f64;
    assert!(hist.recon[last] > floor - 1e-9, "la reconstruccion {} bajo del piso {floor}", hist.recon[last]);
    assert!(
        hist.total(last) - floor < 0.5 * (hist.total(0) - floor),
        "total {} -> {} con piso {floor}",
        hist.total(0),
        hist.total(last)
    );
    assert!(hist.kl[last] > 1.0, "KL final {}", hist.kl[last]);
    assert!(hist.recon.iter().chain(hist.kl.iter()).all(|v| v.is_finite()));
}

#[test]
fn training_is_deterministic_by_seed() {
    let ds = ores();
    let (va, ha) = train_vae(&ds, &config_with(50)).unwrap();
    let (vb, hb) = train_vae(&ds, &config_with(50)).unwrap();
    assert!(weights_equal(&va, &vb));
    assert!(ha.recon.iter().zip(&hb.recon).all(|(p, q)| p.to_bits() == q.to_bits()));
    assert!(ha.kl.iter().zip(&hb.kl).all(|(p, q)| p.to_bits() == q.to_bits()));
    let mut other = config_with(50);
    other.seed = 1;
    let (vc, _) = train_vae(&ds, &other).unwrap();
    assert!(!weights_equal(&va, &vc));
}

#[test]
fn history_has_one_entry_per_epoch_and_total_is_the_sum() {
    let ds = ores();
    let (_, hist) = train_vae(&ds, &config_with(25)).unwrap();
    assert_eq!(hist.recon.len(), 25);
    assert_eq!(hist.kl.len(), 25);
    for i in 0..25 {
        assert_eq!(hist.total(i), hist.recon[i] + hist.kl[i]);
    }
}

#[test]
fn non_finite_loss_is_reported_as_an_error() {
    let ds = ores();
    let mut cfg = config_with(50);
    cfg.optimizer = tp5::optimizer::OptimizerKind::Adam { lr: 1e8, beta1: 0.9, beta2: 0.999, eps: 1e-8 };
    match train_vae(&ds, &cfg) {
        Ok((_, hist)) => {
            assert!(hist.recon.iter().chain(hist.kl.iter()).all(|v| v.is_finite()), "historial con NaN o infinito");
        }
        Err(e) => {
            let msg = e.to_string();
            assert!(msg.contains("epoca"), "el mensaje no nombra la epoca: {msg}");
        }
    }
}

#[test]
fn loss_table_has_header_and_endpoints() {
    let (_, _, hist) = quick();
    let table = render_loss_table(hist);
    let lines: Vec<&str> = table.lines().collect();
    assert_eq!(lines[0], "epoca  reconstruccion  kl  total");
    let rows = &lines[1..];
    assert!(rows.len() <= 12, "{} filas", rows.len());
    let epochs: Vec<usize> = rows.iter().map(|l| l.split_whitespace().next().unwrap().parse().unwrap()).collect();
    assert_eq!(epochs[0], 1);
    assert_eq!(*epochs.last().unwrap(), QUICK_EPOCHS);
    assert!(epochs.windows(2).all(|w| w[0] < w[1]), "{epochs:?}");
}

#[test]
fn loss_summary_line_matches_the_history() {
    let (_, _, hist) = quick();
    let last = QUICK_EPOCHS - 1;
    let expected = format!(
        "PERDIDA: reconstruccion {:.2} -> {:.2} | KL {:.2} -> {:.2} | total {:.2} -> {:.2}",
        hist.recon[0],
        hist.recon[last],
        hist.kl[0],
        hist.kl[last],
        hist.total(0),
        hist.total(last)
    );
    assert_eq!(loss_summary_line(hist), expected);
    assert!(expected.starts_with("PERDIDA: reconstruccion "));
}

#[test]
fn loss_csv_has_header_and_one_row_per_epoch() {
    let (_, _, hist) = quick();
    let dir = unique_tmp_dir("vae-loss-csv");
    let path = dir.join("vae_loss.csv");
    write_loss_csv(&path, hist).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines[0], "epoch,recon,kl,total");
    assert_eq!(lines.len(), QUICK_EPOCHS + 1);
    for l in &lines[1..] {
        assert_eq!(l.matches(',').count(), 3, "{l}");
    }
    assert!(lines[1].starts_with("1,"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn loss_chart_writes_png_with_the_fixed_name() {
    let (_, _, hist) = quick();
    let dir = unique_tmp_dir("vae-loss-chart");
    let path = export_loss_chart(hist, &dir).unwrap();
    assert_eq!(path, dir.join(LOSS_PNG));
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..8], &PNG_SIGNATURE);
    assert!(bytes.len() > 3000);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn validate_epochs_rejects_zero_and_huge() {
    assert!(validate_epochs(0).is_err());
    assert!(validate_epochs(20001).is_err());
    assert_eq!(validate_epochs(1).unwrap(), 1);
    assert_eq!(validate_epochs(20000).unwrap(), 20000);
    assert!(validate_epochs(0).unwrap_err().to_string().contains('0'));
    assert!(validate_epochs(20001).unwrap_err().to_string().contains("20001"));
}

#[test]
fn reconstruction_summary_matches_an_independent_count() {
    let (ds, vae, _) = quick();
    let summary = reconstruction_summary(vae, ds);
    let recon = vae.reconstruct_mean(&ds.x);
    let mut counts = Vec::new();
    let mut level_error = 0.0;
    for i in 0..ds.x.nrows() {
        let mut wrong = 0usize;
        for p in 0..256 {
            if (0..3).any(|c| (ds.x[[i, 3 * p + c]] - recon[[i, 3 * p + c]]).abs() > 32.0 / 255.0) {
                wrong += 1;
            }
        }
        counts.push(wrong);
        level_error += 255.0 * (0..768).map(|k| (ds.x[[i, k]] - recon[[i, k]]).abs()).sum::<f64>() / 768.0;
    }
    assert_eq!(summary.counts, counts);
    let mean = counts.iter().sum::<usize>() as f64 / counts.len() as f64;
    assert!((summary.mean_wrong - mean).abs() < 1e-12);
    assert_eq!(summary.max_wrong, *counts.iter().max().unwrap());
    assert_eq!(summary.n_exact, counts.iter().filter(|&&c| c == 0).count());
    assert!((summary.mean_level_error - level_error / ds.x.nrows() as f64).abs() < 1e-9);
}

#[test]
fn reconstruction_line_format() {
    let (ds, vae, _) = quick();
    let s = reconstruction_summary(vae, ds);
    let expected = format!(
        "RECONSTRUCCION: error medio {:.2} niveles por canal | pixeles incorrectos (algun canal a mas de 32 niveles): media {:.2}, maximo {} de 256 por bloque; {}/18 bloques sin pixeles incorrectos",
        s.mean_level_error, s.mean_wrong, s.max_wrong, s.n_exact
    );
    assert_eq!(reconstruction_line(&s), expected);
}

/// El dataset de menas; falta si no se corrio `bash scripts/descargar_menas.sh`.
fn ores() -> tp5::ores::OreDataset {
    load_ores().expect("dataset de menas: corra `bash scripts/descargar_menas.sh`")
}
