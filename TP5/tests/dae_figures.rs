//! Figuras del Ej 1b: CSV por nivel, graficos contra el nivel de ruido y ejemplos lado a lado.

use std::sync::OnceLock;

use tp5::autoencoder::Autoencoder;
use tp5::dae_figures::{
    build_dae_example_rows, build_noise_examples, dae_examples_filename, export_dae_examples, export_level_charts,
    write_levels_csv, EXAMPLES_NOISE_SEED, ERROR_CHART_PNG, LEVELS_CSV_HEADER, WITHIN_CHART_PNG,
};
use tp5::data::{load_font, N_PIXELS};
use tp5::denoising::{base_dae_config, train_dae, LevelStats};
use tp5::report::wrong_pixel_counts;

const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

fn unique_tmp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("tp5-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn synthetic_stats() -> Vec<LevelStats> {
    [0.0, 0.1, 0.2, 0.3]
        .iter()
        .enumerate()
        .map(|(i, &level)| LevelStats {
            level,
            n_samples: 3200,
            input_mean_wrong: 35.0 * level,
            output_mean_wrong: 0.5 * i as f64 * i as f64,
            output_max_wrong: 3 * i,
            input_frac_within: if i == 0 { 1.0 } else { 0.1 },
            frac_within: 1.0 - 0.2 * i as f64,
        })
        .collect()
}

#[test]
fn levels_csv_has_header_and_one_row_per_level() {
    let dir = unique_tmp_dir("levels-csv");
    let path = dir.join("dae_levels.csv");
    let stats = synthetic_stats();
    write_levels_csv(&path, &stats).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines[0], LEVELS_CSV_HEADER);
    assert_eq!(lines.len(), stats.len() + 1);
    let commas = LEVELS_CSV_HEADER.matches(',').count();
    for l in &lines[1..] {
        assert_eq!(l.matches(',').count(), commas, "{l}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn level_charts_write_two_pngs_with_fixed_names() {
    let dir = unique_tmp_dir("level-charts");
    let paths = export_level_charts(&synthetic_stats(), &dir).unwrap();
    assert_eq!(paths, vec![dir.join(ERROR_CHART_PNG), dir.join(WITHIN_CHART_PNG)]);
    for p in &paths {
        let bytes = std::fs::read(p).unwrap();
        assert_eq!(&bytes[..8], &PNG_SIGNATURE);
        assert!(bytes.len() > 3000, "{} de {} bytes", p.display(), bytes.len());
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn level_charts_need_at_least_two_levels() {
    let dir = unique_tmp_dir("level-charts-one");
    let one = vec![synthetic_stats().remove(1)];
    assert!(export_level_charts(&one, &dir).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

/// Red entrenada rapido (200 epocas, 2 copias) para probar la estructura de las filas.
fn quick_ae() -> &'static Autoencoder {
    static AE: OnceLock<Autoencoder> = OnceLock::new();
    AE.get_or_init(|| {
        let ds = load_font().unwrap();
        let mut cfg = base_dae_config();
        cfg.epochs = 200;
        cfg.copies = 2;
        train_dae(&ds, &cfg).0
    })
}

#[test]
fn dae_example_rows_have_three_rows_of_eight_cells() {
    let ds = load_font().unwrap();
    let ex = build_noise_examples(&ds, &[0.1, 0.3], EXAMPLES_NOISE_SEED).unwrap();
    let rows = build_dae_example_rows(quick_ae(), &ex, 1);
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0].label, "original");
    assert_eq!(rows[1].label, "entrada ruidosa p=0.30");
    assert_eq!(rows[2].label, "salida reconstruida");
    for r in &rows {
        assert_eq!(r.cells.len(), 8);
        assert!(r.cells.iter().all(|(_, p)| p.len() == N_PIXELS));
    }
    let captions: Vec<&str> = rows[0].cells.iter().map(|(c, _)| c.as_str()).collect();
    assert_eq!(captions, ["a", "b", "e", "g", "k", "s", "w", "z"]);
}

#[test]
fn dae_example_noisy_row_equals_the_noise_examples_draw() {
    let ds = load_font().unwrap();
    let levels = [0.05, 0.1, 0.2];
    let ex = build_noise_examples(&ds, &levels, EXAMPLES_NOISE_SEED).unwrap();
    let again = build_noise_examples(&ds, &levels, EXAMPLES_NOISE_SEED).unwrap();
    for i in 0..levels.len() {
        let rows = build_dae_example_rows(quick_ae(), &ex, i);
        for (k, (_, p)) in rows[1].cells.iter().enumerate() {
            assert_eq!(p, &ex.noisy[i].row(k));
            assert_eq!(p, &again.noisy[i].row(k));
        }
    }
}

#[test]
fn dae_example_captions_match_an_independent_pixel_count() {
    let ds = load_font().unwrap();
    let ex = build_noise_examples(&ds, &[0.2], EXAMPLES_NOISE_SEED).unwrap();
    let rows = build_dae_example_rows(quick_ae(), &ex, 0);
    let recon = quick_ae().reconstruct(&ex.noisy[0]);
    for k in 0..8 {
        let noisy_wrong = (0..N_PIXELS).filter(|&p| (ex.clean[[k, p]] >= 0.5) != (ex.noisy[0][[k, p]] >= 0.5)).count();
        let recon_wrong = (0..N_PIXELS).filter(|&p| (ex.clean[[k, p]] >= 0.5) != (recon[[k, p]] >= 0.5)).count();
        assert_eq!(rows[1].cells[k].0, format!("{noisy_wrong} px"));
        assert_eq!(rows[2].cells[k].0, format!("{recon_wrong} px"));
    }
    let counts = wrong_pixel_counts(&ex.clean, &ex.noisy[0]);
    assert_eq!(rows[1].cells[0].0, format!("{} px", counts[0]));
}

#[test]
fn level_zero_noisy_row_equals_the_original_row() {
    let ds = load_font().unwrap();
    let ex = build_noise_examples(&ds, &[0.0], EXAMPLES_NOISE_SEED).unwrap();
    let rows = build_dae_example_rows(quick_ae(), &ex, 0);
    for k in 0..8 {
        assert_eq!(rows[0].cells[k].1, rows[1].cells[k].1);
        assert_eq!(rows[1].cells[k].0, "0 px");
    }
}

#[test]
fn dae_examples_write_one_png_per_level_with_fixed_names() {
    let ds = load_font().unwrap();
    let ex = build_noise_examples(&ds, &[0.0, 0.1, 0.3], EXAMPLES_NOISE_SEED).unwrap();
    let dir = unique_tmp_dir("dae-examples");
    let paths = export_dae_examples(quick_ae(), &ex, &dir).unwrap();
    let names = ["dae_examples_p0.00.png", "dae_examples_p0.10.png", "dae_examples_p0.30.png"];
    assert_eq!(paths.len(), 3);
    for (p, n) in paths.iter().zip(names) {
        assert_eq!(p, &dir.join(n));
        let bytes = std::fs::read(p).unwrap();
        assert_eq!(&bytes[..8], &PNG_SIGNATURE);
    }
    assert_eq!(dae_examples_filename(0.1), "dae_examples_p0.10.png");
    let _ = std::fs::remove_dir_all(&dir);
}
