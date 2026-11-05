//! Grilla, determinismo, puntuacion, CSV, comparacion y regla de seleccion del estudio del DAE.

use std::collections::HashSet;

use tp5::dae_experiments::{
    dae_sweep_grid, dae_variants_of, is_eligible, run_dae_experiment, run_dae_sweep, score_of, write_dae_sweep_csv,
    DAE_STUDIES, SELECTION_NOISE_SEED, SWEEP_CSV_HEADER, SWEEP_LEVELS, SWEEP_SAMPLES,
};
use tp5::data::load_font;
use tp5::denoising::{evaluate_levels, LevelStats, TEST_NOISE_SEED};

fn unique_tmp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("tp5-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn stat(level: f64, output_mean_wrong: f64, frac_within: f64) -> LevelStats {
    LevelStats {
        level,
        n_samples: 3200,
        input_mean_wrong: 0.0,
        output_mean_wrong,
        output_max_wrong: 0,
        input_frac_within: 0.0,
        frac_within,
    }
}

#[test]
fn full_grid_has_39_unique_valid_names_and_13_variants() {
    let grid = dae_sweep_grid(false);
    assert_eq!(grid.len(), 39);
    let names: HashSet<&str> = grid.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names.len(), 39, "nombres repetidos");
    for c in &grid {
        assert!(
            c.name.chars().all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '.' || ch == '-'),
            "nombre invalido: {}",
            c.name
        );
    }
    let variants: HashSet<String> = grid.iter().map(|c| c.variant_key()).collect();
    assert_eq!(variants.len(), 13);
    let counts: Vec<usize> = DAE_STUDIES
        .iter()
        .map(|s| {
            let keys: HashSet<String> = grid.iter().filter(|c| c.study == *s).map(|c| c.variant_key()).collect();
            keys.len()
        })
        .collect();
    assert_eq!(counts, vec![4, 4, 5]);
}

#[test]
fn quick_grid_has_13_configs_with_one_seed() {
    let grid = dae_sweep_grid(true);
    assert_eq!(grid.len(), 13);
    assert!(grid.iter().all(|c| c.seed == 0 && c.epochs == 200 && c.copies == 2));
}

#[test]
fn every_grid_config_has_a_valid_noise_schedule_and_known_study() {
    for quick in [true, false] {
        for c in dae_sweep_grid(quick) {
            assert!(c.train_noise.validate().is_ok(), "ruido invalido en {}", c.name);
            assert!(DAE_STUDIES.contains(&c.study.as_str()), "estudio desconocido en {}", c.name);
            assert!(c.spec.latent_dim >= 1);
        }
    }
}

#[test]
fn quick_run_is_deterministic() {
    let ds = load_font().unwrap();
    let cfg = dae_sweep_grid(true).remove(0);
    let (_, a) = run_dae_experiment(&ds, &cfg);
    let (_, b) = run_dae_experiment(&ds, &cfg);
    assert_eq!(a.score.to_bits(), b.score.to_bits());
    assert_eq!(a.history.tail_loss.to_bits(), b.history.tail_loss.to_bits());
}

#[test]
fn parallel_equals_sequential() {
    let ds = load_font().unwrap();
    let configs: Vec<_> = dae_sweep_grid(true).into_iter().take(3).collect();
    let seq = run_dae_sweep(&ds, &configs, 1);
    let par = run_dae_sweep(&ds, &configs, 4);
    assert_eq!(seq.len(), 3);
    assert_eq!(par.len(), 3);
    for (a, b) in seq.iter().zip(&par) {
        assert_eq!(a.config.name, b.config.name);
        assert_eq!(a.score.to_bits(), b.score.to_bits());
    }
    assert!(run_dae_sweep(&ds, &[], 4).is_empty());
}

#[test]
fn scores_use_the_selection_noise_set() {
    assert_ne!(SELECTION_NOISE_SEED, TEST_NOISE_SEED);
    let ds = load_font().unwrap();
    let cfg = dae_sweep_grid(true).remove(0);
    let (ae, res) = run_dae_experiment(&ds, &cfg);
    let again = evaluate_levels(&ae, &ds.x, &SWEEP_LEVELS, SWEEP_SAMPLES, SELECTION_NOISE_SEED);
    assert_eq!(again, res.stats);
    let on_test = evaluate_levels(&ae, &ds.x, &SWEEP_LEVELS, SWEEP_SAMPLES, TEST_NOISE_SEED);
    assert_ne!(on_test, res.stats, "el ruido de prueba y el de seleccion deben ser distintos");
}

#[test]
fn csv_has_header_and_one_row_per_result() {
    let ds = load_font().unwrap();
    let configs: Vec<_> = dae_sweep_grid(true).into_iter().take(3).collect();
    let results = run_dae_sweep(&ds, &configs, 2);
    let dir = unique_tmp_dir("dae-csv");
    let path = dir.join("dae_sweep.csv");
    write_dae_sweep_csv(&path, &results).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 4);
    assert_eq!(lines[0], SWEEP_CSV_HEADER);
    let commas = SWEEP_CSV_HEADER.matches(',').count();
    for l in &lines[1..] {
        assert_eq!(l.matches(',').count(), commas, "fila con otra cantidad de columnas: {l}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn score_is_the_mean_over_positive_levels() {
    let stats = vec![
        stat(0.0, 9.0, 1.0),
        stat(0.05, 1.0, 1.0),
        stat(0.10, 2.0, 1.0),
        stat(0.15, 3.0, 1.0),
        stat(0.20, 4.0, 1.0),
        stat(0.30, 5.0, 1.0),
    ];
    assert_eq!(score_of(&stats), 3.0);
}

#[test]
fn eligibility_requires_all_clean_patterns_within_tolerance() {
    assert!(is_eligible(&[stat(0.0, 0.0, 1.0), stat(0.1, 1.0, 0.5)]));
    assert!(!is_eligible(&[stat(0.0, 0.0, 0.96875)]));
    assert!(!is_eligible(&[stat(0.05, 0.0, 1.0)]));
    assert!(dae_variants_of(&[], "lat").is_empty());
}
