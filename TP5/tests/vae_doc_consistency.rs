//! T-03-06: los numeros de docs/ej2-analisis.md coinciden con lo que produce el codigo.

use std::sync::OnceLock;

use tp5::ej2::{dataset_line, run_ej2, separation_line, Ej2Options, Ej2Run};
use tp5::emoji::format_emojis_side_by_side;
use tp5::vae::{config_line, loss_summary_line, reconstruction_line, render_loss_table};
use tp5::vae_eval::{contrast_line, manifold_line, new_emoji_line, prior_line, render_acceptance};

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
        "## Dataset de emojis",
        "## Arquitectura y esquema variacional",
        "## Evolución de la pérdida",
        "## Reconstrucción y espacio latente",
        "## Generación de un emoji nuevo",
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
fn document_contains_the_new_emoji_and_prior_lines() {
    let run = run();
    assert_contains("la linea EMOJI NUEVO", &new_emoji_line(&run.search, &run.ds, &run.sep));
    assert_contains("la linea PRIOR del VAE", &prior_line(&run.prior, &run.sep));
    assert_contains("la linea PRIOR del autoencoder simple", &prior_line(&run.plain_prior, &run.sep));
}

#[test]
fn document_contains_every_acceptance_line() {
    let run = run();
    for line in render_acceptance(&run.checks).lines() {
        assert_contains("una linea de aceptacion", line);
    }
    let n = doc().lines().filter(|l| l.starts_with("ACEPTACION OK:")).count();
    assert_eq!(n, 7, "el documento debe tener exactamente 7 lineas ACEPTACION OK:");
}

#[test]
fn document_contains_the_headline_emoji_rendering() {
    let run = run();
    let first = run.search.accepted.first().expect("la corrida acepto un emoji nuevo");
    let nearest = run.ds.x.row(first.judgement.nearest_index).to_owned();
    let binarized = first.image.mapv(|v| if v >= 0.5 { 1.0 } else { 0.0 });
    assert_contains("el dibujo del emoji nuevo junto al mas cercano", &format_emojis_side_by_side(&[binarized, nearest]));
}

#[test]
fn document_cites_generated_artifacts() {
    for path in [
        "output/ej2/emojis.png",
        "output/ej2/vae_loss.csv",
        "output/ej2/vae_loss.png",
        "output/ej2/vae_reconstruction.png",
        "output/ej2/vae_manifold.png",
        "output/ej2/vae_new_emoji.png",
        "output/ej2/plain_manifold.png",
    ] {
        assert_contains("una figura o dato citado", path);
    }
}
