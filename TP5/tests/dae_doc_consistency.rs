//! T-02-05: los numeros de docs/ej1b-analisis.md coinciden con lo que produce el codigo.

use std::sync::OnceLock;

use tp5::autoencoder::Autoencoder;
use tp5::dae_experiments::{
    chosen_dae_config, dae_sweep_grid, render_dae_comparison_md, run_dae_sweep, select_dae_variant, selection_line,
    train_chosen_dae, DaeResult,
};
use tp5::dae_figures::dae_examples_filename;
use tp5::data::load_font;
use tp5::denoising::{
    evaluate_levels, max_level_line, render_level_table, DEFAULT_LEVELS, N_EVAL_SAMPLES, TEST_NOISE_SEED,
};

fn doc() -> &'static str {
    static DOC: OnceLock<String> = OnceLock::new();
    DOC.get_or_init(|| {
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/ej1b-analisis.md"))
            .expect("docs/ej1b-analisis.md")
    })
}

/// Barrido completo (13 variantes x 3 semillas) con 8 hilos, compartido por las pruebas.
fn full_sweep() -> &'static Vec<DaeResult> {
    static SWEEP: OnceLock<Vec<DaeResult>> = OnceLock::new();
    SWEEP.get_or_init(|| {
        let ds = load_font().unwrap();
        run_dae_sweep(&ds, &dae_sweep_grid(false), 8)
    })
}

/// La red elegida entrenada como lo hace `ej1b`.
fn chosen_network() -> &'static Autoencoder {
    static AE: OnceLock<Autoencoder> = OnceLock::new();
    AE.get_or_init(|| {
        let ds = load_font().unwrap();
        train_chosen_dae(&ds).0
    })
}

#[test]
fn document_has_the_six_sections() {
    let headings = [
        "## Arquitectura elegida y justificación",
        "## Estudio de arquitecturas y ruido de entrenamiento",
        "## Niveles de ruido y distorsiones",
        "## Eliminación de ruido por nivel",
        "## Ejemplos visuales",
        "## Alcance y limitaciones",
    ];
    let mut last = 0;
    for h in headings {
        let pos = doc().find(h).unwrap_or_else(|| panic!("falta la seccion '{h}'"));
        assert!(pos >= last, "la seccion '{h}' esta fuera de orden");
        last = pos;
    }
    let level2 = doc().lines().filter(|l| l.starts_with("## ")).count();
    assert_eq!(level2, headings.len());
}

#[test]
fn document_lists_every_tried_variant() {
    for cfg in dae_sweep_grid(false) {
        let key = cfg.variant_key();
        assert!(doc().contains(&key), "la variante '{key}' no aparece en el documento");
    }
}

#[test]
fn document_names_the_chosen_configuration() {
    assert!(doc().contains(&chosen_dae_config().name));
}

#[test]
fn document_embeds_the_comparison_rows_of_the_full_sweep() {
    let md = render_dae_comparison_md(full_sweep());
    let rows: Vec<&str> = md.lines().filter(|l| l.starts_with("| ")).collect();
    assert!(rows.len() >= 13, "el barrido completo debe tener al menos 13 filas de variantes");
    for row in rows {
        assert!(doc().contains(row), "el documento no contiene la fila '{row}'");
    }
}

#[test]
fn document_states_the_selection_line() {
    let line = selection_line(select_dae_variant(full_sweep()).as_ref());
    assert!(doc().contains(&line), "el documento no contiene '{line}'");
}

#[test]
fn document_contains_the_level_table_and_headline() {
    let ds = load_font().unwrap();
    let stats = evaluate_levels(chosen_network(), &ds.x, &DEFAULT_LEVELS, N_EVAL_SAMPLES, TEST_NOISE_SEED);
    let table = render_level_table(&stats);
    assert!(doc().contains(&table), "el documento no contiene la tabla por nivel:\n{table}");
    let line = max_level_line(&stats);
    assert!(doc().contains(&line), "el documento no contiene '{line}'");
}

#[test]
fn document_cites_generated_artifacts() {
    let mut cited = vec![
        "output/ej1b/dae_sweep.csv".to_string(),
        "output/ej1b/dae_comparison.md".to_string(),
        "output/ej1b/dae_levels.csv".to_string(),
        "output/ej1b/noise_examples.png".to_string(),
        "output/ej1b/dae_error_vs_noise.png".to_string(),
        "output/ej1b/dae_within_vs_noise.png".to_string(),
    ];
    for study in ["lat", "hid", "noise"] {
        cited.push(format!("output/ej1b/dae_curves_{study}_loss.png"));
    }
    for level in DEFAULT_LEVELS {
        cited.push(format!("output/ej1b/{}", dae_examples_filename(level)));
    }
    for c in cited {
        assert!(doc().contains(&c), "el documento no cita '{c}'");
    }
}
