//! T-04-03: el README y scripts/regenerar.sh coinciden con lo que escribe el codigo y con los
//! subcomandos de la linea de comandos.

use std::process::Command;

use tp5::analysis::{LATENT_CSV, LATENT_PNG};
use tp5::dae_experiments::DAE_STUDIES;
use tp5::dae_figures::{
    dae_examples_filename, ERROR_CHART_PNG, LEVELS_CSV, NOISE_EXAMPLES_PNG, WITHIN_CHART_PNG,
};
use tp5::denoising::DEFAULT_LEVELS;
use tp5::experiments::STUDIES;
use tp5::generate::NEW_LETTER_PNG;
use tp5::vae_figures::{
    EMOJIS_PNG, LOSS_CSV, LOSS_PNG, MANIFOLD_PNG, NEW_EMOJI_PNG, PLAIN_MANIFOLD_PNG, RECON_PNG,
};

/// Subcomandos de la CLI que el README documenta (seccion "Subcomandos").
const SUBCOMMANDS: [&str; 15] = [
    "show",
    "smoke",
    "sweep",
    "ej1a",
    "ej1b",
    "dae-sweep",
    "noise",
    "latent",
    "generate",
    "emojis",
    "vae",
    "vae-sample",
    "vae-manifold",
    "vae-baseline",
    "ej2",
];

fn read(relative: &str) -> String {
    std::fs::read_to_string(format!("{}/{relative}", env!("CARGO_MANIFEST_DIR")))
        .unwrap_or_else(|e| panic!("no se pudo leer {relative}: {e}"))
}

fn readme() -> String {
    read("README.md")
}

#[test]
fn readme_has_the_required_sections() {
    let readme = readme();
    let headings = [
        "## Qué es",
        "## Requisitos",
        "## Compilar desde un checkout limpio",
        "## Reproducir cada ejercicio",
        "## Subcomandos",
        "## Pruebas",
        "## Documentos de análisis",
        "## Verificar el checkout limpio",
    ];
    // No se cuenta el total de titulos de nivel 2: planes posteriores agregan secciones.
    let found: Vec<&str> = readme.lines().filter(|l| l.starts_with("## ")).collect();
    let mut last = 0;
    for heading in headings {
        let pos = found
            .iter()
            .position(|l| *l == heading)
            .unwrap_or_else(|| panic!("el README no tiene el titulo `{heading}`"));
        assert!(pos >= last, "el titulo `{heading}` esta fuera de orden");
        last = pos;
    }
}

#[test]
fn readme_documents_one_command_per_exercise() {
    let readme = readme();
    for exercise in ["1a", "1b", "2", "todo"] {
        let command = format!("bash scripts/regenerar.sh {exercise}");
        assert!(readme.contains(&command), "el README no contiene `{command}`");
    }
}

#[test]
fn readme_documents_the_locked_clean_build() {
    assert!(readme().contains("cargo build --release --locked"), "el README no contiene la compilacion con --locked");
}

#[test]
fn scripts_exist_and_are_named_in_the_readme() {
    let readme = readme();
    for script in ["scripts/regenerar.sh", "scripts/verificar_checkout_limpio.sh"] {
        let text = read(script);
        assert!(!text.contains('\r'), "{script} tiene retornos de carro: bash en WSL no lo ejecuta");
        assert!(readme.contains(script), "el README no nombra {script}");
    }
}

/// Nombres de los archivos que escribe cada ejercicio, derivados de las constantes de la
/// biblioteca. Los nombres `comparison.md`, `sweep.csv`, `report.csv`, `dae_sweep.csv` y
/// `dae_comparison.md` son constantes privadas de src/main.rs y por eso se escriben literales.
fn artifacts() -> Vec<(&'static str, Vec<String>)> {
    let ej1a: Vec<String> = ["comparison.md", "sweep.csv", "report.csv"]
        .iter()
        .map(|s| s.to_string())
        .chain([LATENT_PNG, LATENT_CSV, NEW_LETTER_PNG].iter().map(|s| s.to_string()))
        .chain(STUDIES.iter().flat_map(|s| [format!("curves_{s}_loss.png"), format!("curves_{s}_wrongpx.png")]))
        .chain(["salida-sweep.txt", "salida-ej1a.txt"].iter().map(|s| s.to_string()))
        .collect();
    let ej1b: Vec<String> = ["dae_sweep.csv", "dae_comparison.md"]
        .iter()
        .map(|s| s.to_string())
        .chain([NOISE_EXAMPLES_PNG, LEVELS_CSV, ERROR_CHART_PNG, WITHIN_CHART_PNG].iter().map(|s| s.to_string()))
        .chain(DAE_STUDIES.iter().map(|s| format!("dae_curves_{s}_loss.png")))
        .chain(DEFAULT_LEVELS.iter().map(|&l| dae_examples_filename(l)))
        .chain(["salida-dae-sweep.txt", "salida-ej1b.txt"].iter().map(|s| s.to_string()))
        .collect();
    let ej2: Vec<String> = [EMOJIS_PNG, RECON_PNG, LOSS_CSV, LOSS_PNG, NEW_EMOJI_PNG, MANIFOLD_PNG, PLAIN_MANIFOLD_PNG]
        .iter()
        .map(|s| s.to_string())
        .chain(std::iter::once("salida-ej2.txt".to_string()))
        .collect();
    vec![("1a", ej1a), ("1b", ej1b), ("2", ej2)]
}

#[test]
fn readme_and_script_name_every_artifact_the_code_writes() {
    let readme = readme();
    let script = read("scripts/regenerar.sh");
    let expected_counts = [("1a", 16), ("1b", 19), ("2", 8)];
    for (exercise, names) in artifacts() {
        let expected = expected_counts.iter().find(|(e, _)| *e == exercise).unwrap().1;
        assert_eq!(names.len(), expected, "el ejercicio {exercise} debe escribir {expected} archivos");
        for name in names {
            assert!(readme.contains(&name), "el README no nombra {name} (ejercicio {exercise})");
            assert!(script.contains(&name), "scripts/regenerar.sh no nombra {name} (ejercicio {exercise})");
        }
    }
}

#[test]
fn readme_subcommands_exist_in_the_cli() {
    let output = Command::new(env!("CARGO_BIN_EXE_tp5")).arg("--help").output().expect("ejecutar tp5 --help");
    assert!(output.status.success(), "tp5 --help fallo");
    let help = String::from_utf8_lossy(&output.stdout);
    let listed: Vec<&str> = help.lines().filter_map(|l| l.split_whitespace().next()).collect();
    let readme = readme();
    for name in SUBCOMMANDS {
        assert!(listed.contains(&name), "tp5 --help no lista el subcomando `{name}`:\n{help}");
        let form = format!("cargo run --release -- {name}");
        assert!(readme.contains(&form), "el README no documenta `{form}`");
    }
}

#[test]
fn readme_documents_the_presentation_command() {
    let readme = readme();
    for expected in [
        "## Presentación",
        "bash scripts/presentacion.sh",
        "--salida RAIZ",
        "--destino ARCHIVO",
        "--sin-regenerar",
        "presentacion/TP5-autoencoders.pptx",
        "cargo test --release --test deck_consistency",
    ] {
        assert!(readme.contains(expected), "el README no contiene `{expected}`");
    }
    let script = read("scripts/presentacion.sh");
    assert!(!script.contains('\r'), "scripts/presentacion.sh tiene retornos de carro: bash en WSL no lo ejecuta");
    assert!(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("presentacion/TP5-autoencoders.pptx").is_file(),
        "falta la presentacion versionada presentacion/TP5-autoencoders.pptx"
    );
}

#[test]
fn presentation_section_follows_the_clean_checkout_section() {
    let readme = readme();
    let clean = readme.find("## Verificar el checkout limpio").expect("el README no tiene `## Verificar el checkout limpio`");
    let presentation = readme.find("## Presentación").expect("el README no tiene `## Presentación`");
    assert!(presentation > clean, "`## Presentación` debe ir despues de `## Verificar el checkout limpio`");
}

#[test]
fn readme_names_every_presentation_section_module() {
    let readme = readme();
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("presentacion/secciones");
    let mut modules: Vec<String> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("no se pudo leer {}: {e}", dir.display()))
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.starts_with("seccion_") && name.ends_with(".py"))
        .collect();
    modules.sort();
    assert!(!modules.is_empty(), "no hay modulos de seccion en {}", dir.display());
    for module in modules {
        assert!(readme.contains(&module), "el README no nombra el modulo de seccion {module}");
    }
}
