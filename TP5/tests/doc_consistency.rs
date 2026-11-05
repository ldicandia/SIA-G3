//! T-01-07: los numeros de docs/ej1a-analisis.md coinciden con lo que produce el codigo.

use std::sync::OnceLock;

use tp5::data::load_font;
use tp5::experiments::{chosen_config, sweep_grid, train_chosen, ExperimentResult};
use tp5::generate::{pick_new_letter, scan_novel};

fn doc() -> &'static str {
    static DOC: OnceLock<String> = OnceLock::new();
    DOC.get_or_init(|| {
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/ej1a-analisis.md"))
            .expect("docs/ej1a-analisis.md")
    })
}

struct Trained {
    result: ExperimentResult,
    coords: String,
}

fn trained() -> &'static Trained {
    static CELL: OnceLock<Trained> = OnceLock::new();
    CELL.get_or_init(|| {
        let ds = load_font().unwrap();
        let (ae, result) = train_chosen(&ds);
        let cands = scan_novel(&ae, &ds, 81, 0.1);
        let l = pick_new_letter(&cands).expect("ninguna letra nueva");
        Trained { result, coords: format!("({:.4}, {:.4})", l.z[0], l.z[1]) }
    })
}

#[test]
fn document_has_the_six_sections() {
    let headings = [
        "## Arquitecturas y técnicas probadas",
        "## Comparación de resultados",
        "## Configuración elegida y justificación",
        "## Subconjunto aprendido (AE-04)",
        "## Espacio latente",
        "## Letra nueva",
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
    for cfg in sweep_grid(false) {
        let key = cfg.variant_key();
        assert!(doc().contains(&key), "la variante '{key}' no aparece en el documento");
    }
}

#[test]
fn document_names_the_chosen_configuration() {
    assert!(doc().contains(&chosen_config().name));
}

#[test]
fn document_result_line_matches_the_code() {
    let line = trained().result.report.result_line();
    assert!(doc().contains(&line), "el documento no contiene '{line}'");
}

#[test]
fn document_new_letter_coordinates_match_the_code() {
    let coords = &trained().coords;
    assert!(doc().contains(coords.as_str()), "el documento no contiene las coordenadas {coords}");
}

#[test]
fn document_cites_generated_artifacts() {
    for path in [
        "output/ej1a/latent.png",
        "output/ej1a/new_letter.png",
        "output/ej1a/comparison.md",
        "output/ej1a/sweep.csv",
    ] {
        assert!(doc().contains(path), "el documento no cita {path}");
    }
}
