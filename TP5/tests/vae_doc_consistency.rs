//! T-03-06: los numeros de docs/ej2-analisis.md coinciden con lo que produce el codigo.

use std::sync::OnceLock;

use tp5::ej2::{dataset_line, run_ej2, separation_line, Ej2Options, Ej2Run};
use tp5::vae::{config_line, loss_summary_line, reconstruction_line, render_loss_table};
use tp5::vae_eval::{
    contrast_line, control_line, judge_line, manifold_line, new_ore_line, prior_line, render_acceptance,
    render_gallery,
};

/// La oracion exacta que el documento debe contener (en una sola linea).
const COMBINATIONS_SENTENCE: &str = "El dataset tiene 18 de las 27 combinaciones posibles de 9 minerales por 3 rocas: faltan los 7 minerales que no aparecen en el Nether y el cuarzo en piedra y en pizarra.";

fn doc() -> &'static str {
    static DOC: OnceLock<String> = OnceLock::new();
    DOC.get_or_init(|| {
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/ej2-analisis.md"))
            .expect("docs/ej2-analisis.md")
    })
}

/// La corrida completa con las opciones por defecto, la misma que `cargo run --release -- ej2`.
fn run() -> &'static Ej2Run {
    static RUN: OnceLock<Ej2Run> = OnceLock::new();
    RUN.get_or_init(|| run_ej2(&Ej2Options::default()).expect("corrida completa del Ej 2"))
}

fn assert_contains(what: &str, text: &str) {
    assert!(doc().contains(text), "el documento no contiene {what}:\n{text}");
}

#[test]
fn document_has_the_six_sections() {
    let headings = [
        "## Dataset de bloques",
        "## Arquitectura y esquema variacional",
        "## Evolución de la pérdida",
        "## Reconstrucción y espacio latente",
        "## Generación de un bloque nuevo",
        "## Alcance y limitaciones",
    ];
    let found: Vec<&str> = doc().lines().filter(|l| l.starts_with("## ")).collect();
    assert_eq!(found, headings, "los titulos de nivel 2 deben ser exactamente estos seis y en este orden");
}

#[test]
fn document_states_the_dataset_facts() {
    let run = run();
    assert_contains("la linea del dataset", &dataset_line());
    assert_contains("la linea de separacion", &separation_line(&run.sep));
}

#[test]
fn document_names_the_configuration() {
    let run = run();
    assert_contains("la linea de configuracion", &config_line(&run.cfg, &run.vae));
}

#[test]
fn document_contains_the_loss_table_and_summary() {
    let run = run();
    assert_contains("la tabla de perdida", &render_loss_table(&run.history));
    assert_contains("el resumen de perdida", &loss_summary_line(&run.history));
}

#[test]
fn document_contains_reconstruction_manifold_and_contrast_lines() {
    let run = run();
    assert_contains("la linea RECONSTRUCCION", &reconstruction_line(&run.recon));
    assert_contains("la linea MANIFOLD", &manifold_line(&run.grid, &run.sep));
    assert_contains("la linea CONTRASTE", &contrast_line(&run.contrast));
}

#[test]
fn document_contains_the_prior_lines() {
    let run = run();
    assert_contains("la linea PRIOR del VAE", &format!("VAE {}", prior_line(&run.prior)));
    assert_contains("la linea PRIOR del autoencoder simple", &format!("AE simple {}", prior_line(&run.plain_prior)));
}

#[test]
fn document_contains_the_new_ore_lines() {
    let run = run();
    assert_contains("la linea BLOQUE NUEVO", &new_ore_line(&run.search, &run.ds));
    assert_contains("la linea JUEZ", &judge_line(&run.judge));
    for line in render_gallery(&run.search, &run.ds).lines() {
        assert_contains("una linea de la galeria", line);
    }
}

#[test]
fn document_contains_the_negative_control_line() {
    let run = run();
    assert_contains("la linea CONTROL NEGATIVO", &control_line(&run.control));
}

#[test]
fn document_states_which_combinations_are_missing() {
    assert_contains("la oracion de las combinaciones ausentes", COMBINATIONS_SENTENCE);
    assert_contains("la prueba que respalda la oracion", "combinations_present_are_18_of_27");
}

#[test]
fn document_does_not_mention_the_previous_dataset() {
    for obsolete in ["emoji", "Emoji", "no es una copia de ninguno", "**Nuevo:** está a al menos"] {
        assert!(!doc().contains(obsolete), "el documento todavia contiene la afirmacion obsoleta: {obsolete}");
    }
}

#[test]
fn document_contains_every_acceptance_line() {
    let run = run();
    for line in render_acceptance(&run.checks).lines() {
        assert_contains("una linea de aceptacion", line);
    }
    assert_eq!(run.checks.len(), 11, "la corrida debe evaluar 11 verificaciones de aceptacion");
    let n = doc().lines().filter(|l| l.starts_with("ACEPTACION OK:")).count();
    assert_eq!(n, run.checks.len(), "el documento debe tener exactamente {} lineas ACEPTACION OK:", run.checks.len());
}

#[test]
fn document_names_the_headline_ore_and_its_nearest_block() {
    let run = run();
    let first = run.search.accepted.first().expect("la corrida acepto un bloque nuevo");
    assert_contains("la combinacion del bloque principal", &format!("{} (combinacion ausente del dataset)", first.verdict.combination()));
    assert_contains("el bloque mas cercano", &format!("`{}`, está a {:.1} niveles", run.ds.names[first.nearest.0], first.nearest.1));
}

#[test]
fn document_cites_generated_artifacts() {
    for path in [
        "output/ej2/ores.png",
        "output/ej2/vae_loss.csv",
        "output/ej2/vae_loss.png",
        "output/ej2/vae_reconstruction.png",
        "output/ej2/vae_manifold.png",
        "output/ej2/vae_new_ore.png",
        "output/ej2/plain_manifold.png",
    ] {
        assert_contains("una figura o dato citado", path);
    }
}
