//! Grilla, determinismo, ejecucion paralela y escritores del sweep.

use std::collections::HashSet;

use tp5::data::load_font;
use tp5::experiments::{
    render_comparison_md, run_experiment, run_sweep, sweep_grid, write_sweep_csv, SWEEP_CSV_HEADER,
};

fn unique_tmp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("tp5-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

#[test]
fn full_grid_has_54_unique_valid_names() {
    let grid = sweep_grid(false);
    assert_eq!(grid.len(), 54);
    let names: HashSet<&str> = grid.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names.len(), 54, "nombres repetidos");
    for c in &grid {
        assert!(
            c.name.chars().all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '.' || ch == '-'),
            "nombre invalido: {}",
            c.name
        );
    }
    let variants: HashSet<String> = grid.iter().map(|c| c.variant_key()).collect();
    assert_eq!(variants.len(), 18);
}

#[test]
fn quick_grid_has_18_configs() {
    assert_eq!(sweep_grid(true).len(), 18);
}

#[test]
fn quick_run_is_deterministic() {
    let ds = load_font().unwrap();
    let cfg = sweep_grid(true).into_iter().find(|c| c.name == "optim-adam-lr0.005-s0").unwrap();
    let (_, a) = run_experiment(&ds, &cfg);
    let (_, b) = run_experiment(&ds, &cfg);
    assert_eq!(a.final_loss.to_bits(), b.final_loss.to_bits());
    let ca: Vec<usize> = a.report.patterns.iter().map(|p| p.wrong_pixels).collect();
    let cb: Vec<usize> = b.report.patterns.iter().map(|p| p.wrong_pixels).collect();
    assert_eq!(ca, cb);
}

#[test]
fn parallel_equals_sequential() {
    let ds = load_font().unwrap();
    let configs: Vec<_> = sweep_grid(true).into_iter().take(3).collect();
    let seq = run_sweep(&ds, &configs, 1);
    let par = run_sweep(&ds, &configs, 4);
    assert_eq!(seq.len(), 3);
    assert_eq!(par.len(), 3);
    for (s, p) in seq.iter().zip(&par) {
        assert_eq!(s.config.name, p.config.name);
        assert_eq!(s.final_loss.to_bits(), p.final_loss.to_bits());
    }
}

#[test]
fn csv_has_header_and_one_row_per_result() {
    let ds = load_font().unwrap();
    // Incluye una variante con Momentum, cuya etiqueta de optimizador lleva coma.
    let configs: Vec<_> =
        sweep_grid(true).into_iter().filter(|c| c.name.starts_with("optim-mom") || c.name == "loss-mse-s0").collect();
    assert!(configs.len() >= 3);
    let results = run_sweep(&ds, &configs, 2);
    let dir = unique_tmp_dir("csv");
    let path = dir.join("sweep.csv");
    write_sweep_csv(&path, &results).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines[0], SWEEP_CSV_HEADER);
    assert_eq!(lines.len(), results.len() + 1);
    let commas = lines[0].matches(',').count();
    for l in &lines {
        assert_eq!(l.matches(',').count(), commas, "fila con otro numero de columnas: {l}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn comparison_md_lists_every_variant() {
    let ds = load_font().unwrap();
    let configs = sweep_grid(true);
    let results = run_sweep(&ds, &configs, 4);
    let md = render_comparison_md(&results);
    for c in &configs {
        assert!(md.contains(&c.variant_key()), "falta {}", c.variant_key());
    }
    for study in ["arch", "optim", "act", "loss"] {
        assert!(md.contains(&format!("## Estudio: {study}")));
    }
}

#[test]
fn latent_dimension_is_two_in_every_config() {
    for quick in [true, false] {
        for c in sweep_grid(quick) {
            assert_eq!(c.spec.latent_dim, 2, "{}", c.name);
        }
    }
}
