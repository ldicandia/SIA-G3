//! Ejemplos de distorsion del Ej 1b: tabla de filas en PNG y construccion de los ejemplos.

use ndarray::Array1;

use tp5::dae_figures::{
    build_noise_examples, export_noise_examples, format_grids_side_by_side, EXAMPLES_NOISE_SEED, EXAMPLE_LABELS,
    NOISE_EXAMPLES_PNG,
};
use tp5::data::{load_font, N_PIXELS};
use tp5::plots::{plot_pattern_rows, PatternRow};

fn unique_tmp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("tp5-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn row(label: &str, n: usize) -> PatternRow {
    PatternRow {
        label: label.to_string(),
        cells: (0..n)
            .map(|i| (format!("{i} px"), Array1::from_shape_fn(N_PIXELS, |p| ((p + i) % 2) as f64)))
            .collect(),
    }
}

#[test]
fn plot_pattern_rows_writes_png() {
    let dir = unique_tmp_dir("rows");
    let path = dir.join("r.png");
    plot_pattern_rows(&path, "titulo", &[row("original", 4), row("p=0.10", 4)]).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..8], &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
    assert!(bytes.len() > 3000, "PNG de {} bytes", bytes.len());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn plot_pattern_rows_rejects_ragged_or_empty_rows() {
    let dir = unique_tmp_dir("rows-bad");
    let path = dir.join("r.png");
    assert!(plot_pattern_rows(&path, "t", &[]).is_err());
    assert!(plot_pattern_rows(&path, "t", &[row("vacia", 0)]).is_err());
    assert!(plot_pattern_rows(&path, "t", &[row("a", 3), row("b", 4)]).is_err());
    let mut short = row("corta", 2);
    short.cells[1].1 = Array1::zeros(N_PIXELS - 1);
    assert!(plot_pattern_rows(&path, "t", &[short]).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn build_noise_examples_has_expected_shapes() {
    let ds = load_font().unwrap();
    let levels = [0.05, 0.1, 0.2];
    let ex = build_noise_examples(&ds, &levels, EXAMPLES_NOISE_SEED).unwrap();
    assert_eq!(ex.labels.len(), 8);
    assert_eq!(ex.labels, EXAMPLE_LABELS.iter().map(|s| s.to_string()).collect::<Vec<_>>());
    assert_eq!(ex.clean.dim(), (8, 35));
    assert_eq!(ex.noisy.len(), 3);
    assert!(ex.noisy.iter().all(|m| m.dim() == (8, 35)));
    for (k, label) in EXAMPLE_LABELS.iter().enumerate() {
        let i = ds.labels.iter().position(|l| l == label).unwrap();
        assert_eq!(ex.clean.row(k), ds.x.row(i));
    }
    assert!(build_noise_examples(&ds, &[0.7], EXAMPLES_NOISE_SEED).is_err());
}

#[test]
fn level_zero_example_equals_clean() {
    let ds = load_font().unwrap();
    let ex = build_noise_examples(&ds, &[0.0], EXAMPLES_NOISE_SEED).unwrap();
    assert_eq!(ex.noisy[0], ex.clean);
    assert!(ex.inverted_pixels(0).iter().all(|&k| k == 0));
}

#[test]
fn examples_are_deterministic_by_seed() {
    let ds = load_font().unwrap();
    let a = build_noise_examples(&ds, &[0.3], 1).unwrap();
    let b = build_noise_examples(&ds, &[0.3], 1).unwrap();
    let c = build_noise_examples(&ds, &[0.3], 2).unwrap();
    assert_eq!(a.noisy, b.noisy);
    assert_ne!(a.noisy, c.noisy);
}

#[test]
fn format_grids_side_by_side_has_seven_lines_of_equal_width() {
    let ds = load_font().unwrap();
    let pats: Vec<Array1<f64>> = (0..3).map(|i| ds.x.row(i).to_owned()).collect();
    let text = format_grids_side_by_side(&pats);
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 7);
    assert!(lines.iter().all(|l| l.chars().count() == 3 * 5 + 2 * 2), "{text}");
}

#[test]
fn export_noise_examples_writes_png_with_the_fixed_name() {
    let ds = load_font().unwrap();
    let ex = build_noise_examples(&ds, &[0.05, 0.2], EXAMPLES_NOISE_SEED).unwrap();
    let dir = unique_tmp_dir("noise-ex");
    let path = export_noise_examples(&ex, &dir).unwrap();
    assert_eq!(path, dir.join(NOISE_EXAMPLES_PNG));
    assert_eq!(NOISE_EXAMPLES_PNG, "noise_examples.png");
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..8], &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
    let _ = std::fs::remove_dir_all(&dir);
}
