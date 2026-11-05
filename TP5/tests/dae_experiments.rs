//! Grilla, determinismo, puntuacion, CSV, comparacion y regla de seleccion del estudio del DAE.

use std::collections::HashSet;

use tp5::dae_experiments::{
    dae_sweep_grid, dae_variants_of, is_eligible, render_dae_comparison_md, run_dae_experiment, run_dae_sweep,
    score_of, select_dae_variant, selection_line, write_dae_sweep_csv, DaeResult, Selection, DAE_STUDIES,
    SELECTION_NOISE_SEED, SWEEP_CSV_HEADER, SWEEP_LEVELS, SWEEP_SAMPLES,
};
use tp5::data::load_font;
use tp5::denoising::{base_dae_config, evaluate_levels, DaeHistory, LevelStats, TEST_NOISE_SEED};
use tp5::plots::write_dae_study_curves;

/// Resultado sintetico: configuracion base con nombre y semilla elegidos, sin estadisticas.
fn synth(name: &str, seed: u64, score: f64, eligible: bool) -> DaeResult {
    let mut config = base_dae_config();
    config.name = name.to_string();
    config.seed = seed;
    DaeResult {
        config,
        history: DaeHistory { loss: vec![], tail_loss: 0.0 },
        stats: vec![],
        score,
        eligible,
    }
}

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

#[test]
fn comparison_md_lists_every_variant_and_study() {
    let ds = load_font().unwrap();
    let results = run_dae_sweep(&ds, &dae_sweep_grid(true), 8);
    let md = render_dae_comparison_md(&results);
    for study in DAE_STUDIES {
        assert!(md.contains(&format!("## Estudio: {study}")), "falta el estudio {study}");
        for key in dae_variants_of(&results, study) {
            assert!(md.contains(&format!("| {key} |")), "falta la variante {key}");
        }
    }
    assert_eq!(md.matches("## Estudio: ").count(), 3);
}

#[test]
fn selection_picks_lowest_median_score_among_eligible_variants() {
    let results = vec![
        synth("a-s0", 0, 2.0, true),
        synth("a-s1", 1, 4.0, true),
        synth("b-s0", 0, 2.5, true),
        synth("b-s1", 1, 2.6, true),
        synth("c-s0", 0, 0.1, false),
        synth("c-s1", 1, 0.1, true),
    ];
    let sel = select_dae_variant(&results).unwrap();
    assert_eq!(sel.variant, "b");
    assert_eq!(sel.median_score, 2.55);
    assert_eq!((sel.n_eligible, sel.n_runs), (2, 2));
}

#[test]
fn selection_ignores_variants_with_an_ineligible_seed() {
    let results = vec![synth("a-s0", 0, 0.5, true), synth("a-s1", 1, 0.5, false), synth("b-s0", 0, 9.0, true)];
    let sel = select_dae_variant(&results).unwrap();
    assert_eq!(sel.variant, "b");
    assert_eq!(sel.name, "b-s0");
}

#[test]
fn selection_is_none_when_nothing_is_eligible() {
    assert!(select_dae_variant(&[synth("a-s0", 0, 1.0, false)]).is_none());
    assert!(select_dae_variant(&[]).is_none());
}

#[test]
fn selection_breaks_ties_by_grid_order() {
    let results = vec![synth("x-s0", 0, 3.0, true), synth("y-s0", 0, 3.0, true), synth("z-s0", 0, 3.5, true)];
    assert_eq!(select_dae_variant(&results).unwrap().variant, "x");
}

#[test]
fn selection_picks_the_best_seed_inside_the_winner() {
    let results = vec![
        synth("w-s0", 0, 2.0, true),
        synth("w-s1", 1, 1.0, true),
        synth("w-s2", 2, 3.0, true),
        synth("v-s0", 0, 9.0, true),
    ];
    let sel = select_dae_variant(&results).unwrap();
    assert_eq!(sel.variant, "w");
    assert_eq!(sel.name, "w-s1");
    assert_eq!(sel.seed, 1);
    assert_eq!(sel.median_score, 2.0);
    // empate de puntaje dentro de la variante: gana la menor semilla
    let tie = vec![synth("t-s1", 1, 1.0, true), synth("t-s0", 0, 1.0, true)];
    assert_eq!(select_dae_variant(&tie).unwrap().seed, 0);
}

#[test]
fn selection_line_formats_both_outcomes() {
    let sel = Selection {
        variant: "lat-l8".to_string(),
        name: "lat-l8-s1".to_string(),
        seed: 1,
        median_score: 1.23456,
        n_eligible: 3,
        n_runs: 3,
    };
    assert_eq!(
        selection_line(Some(&sel)),
        "SELECCION: lat-l8-s1 (variante lat-l8, puntaje mediano 1.2346, 3/3 semillas elegibles)"
    );
    let none = selection_line(None);
    assert!(none.starts_with("SELECCION: ninguna variante elegible"));
}

#[test]
fn study_curves_write_one_png_per_study() {
    let ds = load_font().unwrap();
    let results = run_dae_sweep(&ds, &dae_sweep_grid(true), 8);
    let dir = unique_tmp_dir("dae-curves");
    std::fs::create_dir_all(&dir).unwrap();
    assert_eq!(write_dae_study_curves(&dir, &results).unwrap(), 3);
    for study in ["lat", "hid", "noise"] {
        let bytes = std::fs::read(dir.join(format!("dae_curves_{study}_loss.png"))).unwrap();
        assert!(bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]), "{study}: no es PNG");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
