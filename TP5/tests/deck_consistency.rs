//! T-04-09: la presentacion versionada (presentacion/TP5-autoencoders.pptx) cita lo que el codigo recalcula.
//!
//! No depende de `output/` ni compara bytes de PNG (dependen de las tipografias del sistema): lee el
//! paquete almacenado de la presentacion, extrae el texto de cada diapositiva y lo compara con lo que la
//! biblioteca vuelve a calcular en este proceso. Es la comprobacion independiente de la maquina;
//! `presentacion/verificar.py` hace la otra, contra los archivos regenerados en la maquina que corre.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use tp5::analysis::analyze;
use tp5::autoencoder::Autoencoder;
use tp5::dae_experiments::{
    chosen_dae_config, dae_sweep_grid, render_dae_comparison_md, run_dae_sweep, select_dae_variant, selection_line,
    train_chosen_dae, DaeResult,
};
use tp5::dae_figures::dae_examples_filename;
use tp5::data::{format_grid, load_font};
use tp5::denoising::{
    evaluate_levels, max_level_line, render_level_table, DEFAULT_LEVELS, N_EVAL_SAMPLES, TEST_NOISE_SEED,
};
use tp5::ej2::{dataset_line, result_line, run_ej2, separation_line, Ej2Options, Ej2Run};
use tp5::emoji::format_emojis_side_by_side;
use tp5::experiments::{chosen_config, render_comparison_md, run_sweep, sweep_grid, train_chosen};
use tp5::generate::{pick_new_letter, scan_novel};
use tp5::vae::{config_line, loss_summary_line, reconstruction_line, render_loss_table};
use tp5::vae_eval::{
    contrast_line, control_line, manifold_line, new_emoji_line, prior_line, regions_line, render_acceptance,
    render_gallery,
};

const DECK_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/presentacion/TP5-autoencoders.pptx");
const PORTADA: &str = "TP5: autoencoders (básico, denoising y variacional)";

/// Las afirmaciones retiradas del primer criterio de novedad (tests/vae_doc_consistency.rs y sus formas
/// genericas): ninguna puede aparecer en el texto de ninguna diapositiva.
const RETRACTED: [&str; 5] = [
    "no es una copia de ninguno",
    "no es una copia",
    "no es copia",
    "no se trata de una copia",
    "está a al menos",
];

/// Unica constante de teoria admitida sin fuente en el codigo: el 0.5 de las formulas del VAE.
const THEORY_CONSTANTS: [&str; 1] = ["0.5"];

// --------------------------------------------------------------------------------------------------
// Lector del paquete almacenado
// --------------------------------------------------------------------------------------------------

fn le16(b: &[u8], at: usize) -> usize {
    u16::from_le_bytes([b[at], b[at + 1]]) as usize
}

fn le32(b: &[u8], at: usize) -> usize {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]) as usize
}

/// Recorre los encabezados locales de un ZIP con miembros almacenados y devuelve (nombre, bytes) en el
/// orden del archivo. Se detiene en el directorio central. Falla con un mensaje claro ante un miembro
/// comprimido o con descriptor de datos.
fn read_stored_package(bytes: &[u8]) -> Vec<(String, Vec<u8>)> {
    const LOCAL: usize = 0x0403_4b50;
    const CENTRAL: usize = 0x0201_4b50;
    const END: usize = 0x0605_4b50;
    let mut members = Vec::new();
    let mut pos = 0;
    while pos + 4 <= bytes.len() {
        let signature = le32(bytes, pos);
        if signature == CENTRAL || signature == END {
            break;
        }
        assert_eq!(signature, LOCAL, "firma de encabezado local invalida en el desplazamiento {pos}");
        let flags = le16(bytes, pos + 6);
        let method = le16(bytes, pos + 8);
        let compressed = le32(bytes, pos + 18);
        let size = le32(bytes, pos + 22);
        let name_len = le16(bytes, pos + 26);
        let extra_len = le16(bytes, pos + 28);
        let name_at = pos + 30;
        let name = String::from_utf8(bytes[name_at..name_at + name_len].to_vec()).expect("nombre de miembro UTF-8");
        assert_eq!(method, 0, "el miembro {name} esta comprimido (metodo {method}); la presentacion debe estar almacenada");
        assert_eq!(flags & 0x08, 0, "el miembro {name} usa un descriptor de datos; el lector de la prueba no lo admite");
        assert_eq!(compressed, size, "el miembro {name} almacenado tiene tamanos distintos");
        let data_at = name_at + name_len + extra_len;
        assert!(data_at + size <= bytes.len(), "el miembro {name} se sale del archivo");
        members.push((name, bytes[data_at..data_at + size].to_vec()));
        pos = data_at + size;
    }
    members
}

fn unescape(s: &str) -> String {
    s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&apos;", "'").replace("&amp;", "&")
}

struct Slide {
    /// Numero desde 1, el orden de ppt/slides/slideN.xml.
    index: usize,
    title: String,
    /// Todos los tramos de texto (a:t) en el orden del documento, con los espacios iniciales intactos.
    runs: Vec<String>,
    /// Textos alternativos (descr) de las imagenes.
    pictures: Vec<String>,
}

/// Cada tramo `<a:t>...</a:t>` de un documento, con las cinco entidades XML deshechas.
fn text_runs(xml: &str) -> Vec<(usize, String)> {
    let mut runs = Vec::new();
    let mut from = 0;
    while let Some(open) = xml[from..].find("<a:t>") {
        let start = from + open + "<a:t>".len();
        let end = start + xml[start..].find("</a:t>").expect("tramo de texto sin cierre");
        runs.push((from + open, unescape(&xml[start..end])));
        from = end + "</a:t>".len();
    }
    runs
}

fn picture_descriptions(xml: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(open) = xml[from..].find("<p:pic>") {
        let start = from + open;
        let end = start + xml[start..].find("</p:pic>").expect("imagen sin cierre");
        let block = &xml[start..end];
        if let Some(at) = block.find("descr=\"") {
            let from_descr = at + "descr=\"".len();
            let to = from_descr + block[from_descr..].find('"').expect("descr sin cierre");
            out.push(unescape(&block[from_descr..to]));
        }
        from = end;
    }
    out
}

struct Deck {
    member_names: Vec<String>,
    all_stored: bool,
    slides: Vec<Slide>,
}

fn deck() -> &'static Deck {
    static DECK: OnceLock<Deck> = OnceLock::new();
    DECK.get_or_init(|| {
        let bytes = std::fs::read(DECK_PATH).unwrap_or_else(|e| panic!("no se pudo leer {DECK_PATH}: {e}"));
        let members = read_stored_package(&bytes);
        let member_names: Vec<String> = members.iter().map(|(n, _)| n.clone()).collect();
        let by_name: HashMap<&str, &Vec<u8>> = members.iter().map(|(n, d)| (n.as_str(), d)).collect();
        let mut numbers: Vec<usize> = member_names
            .iter()
            .filter_map(|n| n.strip_prefix("ppt/slides/slide").and_then(|r| r.strip_suffix(".xml")))
            .filter_map(|d| d.parse::<usize>().ok())
            .collect();
        numbers.sort_unstable();
        let slides = numbers
            .iter()
            .map(|&n| {
                let xml = String::from_utf8(by_name[format!("ppt/slides/slide{n}.xml").as_str()].clone())
                    .expect("diapositiva UTF-8");
                let runs = text_runs(&xml);
                let marker = xml.find("<p:ph type=\"title\"/>").or_else(|| xml.find("<p:ph type=\"ctrTitle\"/>"));
                let title = marker
                    .and_then(|m| runs.iter().find(|(at, _)| *at > m))
                    .map(|(_, t)| t.clone())
                    .unwrap_or_default();
                Slide {
                    index: n,
                    title,
                    runs: runs.into_iter().map(|(_, t)| t).collect(),
                    pictures: picture_descriptions(&xml),
                }
            })
            .collect();
        Deck { member_names, all_stored: true, slides }
    })
}

impl Deck {
    fn has_run(&self, text: &str) -> bool {
        self.slides.iter().any(|s| s.runs.iter().any(|r| r == text))
    }

    fn has_run_containing(&self, fragment: &str) -> bool {
        self.slides.iter().any(|s| s.runs.iter().any(|r| r.contains(fragment)))
    }

    /// Una fila de tabla: sus celdas, en orden, como tramos consecutivos de una sola diapositiva.
    fn has_row(&self, cells: &[String]) -> bool {
        self.slides.iter().any(|s| s.runs.windows(cells.len()).any(|w| w == cells))
    }
}

fn assert_runs(label: &str, text: &str) {
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        assert!(
            deck().has_run(line),
            "{label}: ninguna de las {} diapositivas tiene como tramo exacto la linea {line:?}",
            deck().slides.len()
        );
    }
}

/// Filas de datos de las tablas Markdown de un texto: las celdas de cada linea posterior al separador.
fn markdown_data_rows(md: &str) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    let mut in_body = false;
    for line in md.lines() {
        let t = line.trim();
        if !t.starts_with('|') {
            in_body = false;
            continue;
        }
        let cells: Vec<String> = t.trim_matches('|').split('|').map(|c| c.trim().to_string()).collect();
        if cells.iter().all(|c| !c.is_empty() && c.chars().all(|ch| ch == '-' || ch == ':')) {
            in_body = true;
            continue;
        }
        if in_body {
            rows.push(cells);
        }
    }
    rows
}

// --------------------------------------------------------------------------------------------------
// Numeros: la misma regla que presentacion/verificar.py
// --------------------------------------------------------------------------------------------------

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Decimales (cifras, punto, cifras) y enteros de tres o mas cifras que no se pegan a una palabra,
/// sin signo. Equivale a la expresion `(?<!\w)(\d+\.\d+|\d{3,})(?!\w)`.
fn numeric_tokens(text: &str) -> Vec<String> {
    let c: Vec<char> = text.chars().collect();
    let digits_from = |mut j: usize| {
        while j < c.len() && c[j].is_ascii_digit() {
            j += 1;
        }
        j
    };
    let mut out = Vec::new();
    let mut i = 0;
    while i < c.len() {
        if !c[i].is_ascii_digit() || (i > 0 && is_word(c[i - 1])) {
            i += 1;
            continue;
        }
        let run_end = digits_from(i);
        let mut matched = None;
        if run_end + 1 < c.len() && c[run_end] == '.' && c[run_end + 1].is_ascii_digit() {
            let end = digits_from(run_end + 1);
            if end == c.len() || !is_word(c[end]) {
                matched = Some(end);
            }
        }
        if matched.is_none() && run_end - i >= 3 && (run_end == c.len() || !is_word(c[run_end])) {
            matched = Some(run_end);
        }
        match matched {
            Some(end) => {
                out.push(c[i..end].iter().collect());
                i = end;
            }
            None => i += 1,
        }
    }
    out
}

// --------------------------------------------------------------------------------------------------
// Recalculo desde la biblioteca (cada cache corre una sola vez)
// --------------------------------------------------------------------------------------------------

struct Ej1aFacts {
    result_line: String,
    config_name: String,
    point_line: String,
    sweep_rows: Vec<Vec<String>>,
    blob: String,
}

fn ej1a() -> &'static Ej1aFacts {
    static CELL: OnceLock<Ej1aFacts> = OnceLock::new();
    CELL.get_or_init(|| {
        let ds = load_font().unwrap();
        let (ae, res) = train_chosen(&ds);
        let cands = scan_novel(&ae, &ds, 81, 0.1);
        let letter = pick_new_letter(&cands).expect("ninguna letra nueva");
        let sweep = run_sweep(&ds, &sweep_grid(false), 8);
        let comparison = render_comparison_md(&sweep);
        let c = &res.config;
        let codes = ae.encode(&ds.x);
        let analysis = analyze(&ds.labels, &ds.x, &codes, &res.report, &res.history).render();
        let point_line = format!("punto latente: ({:.4}, {:.4})", letter.z[0], letter.z[1]);
        let mut blob = String::new();
        blob += &format!("{} {} {} {} {}\n", c.name, ae.sizes_label(), c.optimizer.label(), c.epochs, c.seed);
        blob += &format!("{}\n{}\n{}\n", res.report.render_table(), res.report.result_line(), analysis);
        for (i, (l, r)) in ds.labels.iter().zip(codes.rows()).enumerate() {
            blob += &format!("{:>3}  {:<8}  {:.4}  {:.4}\n", i, l, r[0], r[1]);
        }
        blob += &format!("{point_line}\n{}\n", format_grid(letter.pattern.view()));
        blob += &format!("{} {} {:.4}\n", letter.nearest_label, letter.min_hamming, letter.latent_distance);
        blob += &comparison;
        Ej1aFacts {
            result_line: res.report.result_line(),
            config_name: chosen_config().name,
            point_line,
            sweep_rows: markdown_data_rows(&comparison),
            blob,
        }
    })
}

struct Ej1bFacts {
    level_table: String,
    max_line: String,
    selection: String,
    config_name: String,
    comparison_rows: Vec<Vec<String>>,
    blob: String,
}

fn ej1b() -> &'static Ej1bFacts {
    static CELL: OnceLock<Ej1bFacts> = OnceLock::new();
    CELL.get_or_init(|| {
        let ds = load_font().unwrap();
        let sweep: Vec<DaeResult> = run_dae_sweep(&ds, &dae_sweep_grid(false), 8);
        let (ae, cfg): (Autoencoder, _) = train_chosen_dae(&ds);
        let stats = evaluate_levels(&ae, &ds.x, &DEFAULT_LEVELS, N_EVAL_SAMPLES, TEST_NOISE_SEED);
        let level_table = render_level_table(&stats);
        let max_line = max_level_line(&stats);
        let selection = selection_line(select_dae_variant(&sweep).as_ref());
        let comparison = render_dae_comparison_md(&sweep);
        let mut blob = String::new();
        blob += &format!(
            "{} {} {} {} {} {} {}\n",
            cfg.name,
            ae.sizes_label(),
            cfg.optimizer.label(),
            cfg.epochs,
            cfg.copies,
            cfg.train_noise.label(),
            cfg.seed
        );
        blob += &format!("{N_EVAL_SAMPLES} {TEST_NOISE_SEED}\n{level_table}\n{max_line}\n{selection}\n{comparison}");
        Ej1bFacts {
            level_table,
            max_line,
            selection,
            config_name: chosen_dae_config().name,
            comparison_rows: markdown_data_rows(&comparison),
            blob,
        }
    })
}

fn ej2_run() -> &'static Ej2Run {
    static RUN: OnceLock<Ej2Run> = OnceLock::new();
    RUN.get_or_init(|| run_ej2(&Ej2Options::default()).expect("corrida completa del Ej 2"))
}

/// Cada bloque de texto que el reporte del Ej 2 imprime y que la presentacion cita, con su nombre.
fn ej2_texts() -> &'static Vec<(&'static str, String)> {
    static CELL: OnceLock<Vec<(&'static str, String)>> = OnceLock::new();
    CELL.get_or_init(|| {
        let run = ej2_run();
        let first = run.search.accepted.first().expect("la corrida acepto un emoji nuevo");
        let nearest = run.ds.x.row(first.judgement.nearest_index).to_owned();
        let binarized = first.image.mapv(|v| if v >= 0.5 { 1.0 } else { 0.0 });
        vec![
            ("linea del dataset", dataset_line()),
            ("linea de separacion", separation_line(&run.sep)),
            ("linea de configuracion", config_line(&run.cfg, &run.vae)),
            ("tabla de perdida", render_loss_table(&run.history)),
            ("resumen de perdida", loss_summary_line(&run.history)),
            ("linea RECONSTRUCCION", reconstruction_line(&run.recon)),
            ("linea MANIFOLD", manifold_line(&run.grid, &run.sep)),
            ("linea CONTRASTE", contrast_line(&run.contrast)),
            ("linea PRIOR del VAE", format!("VAE {}", prior_line(&run.prior, &run.sep))),
            ("linea PRIOR del autoencoder simple", format!("AE simple {}", prior_line(&run.plain_prior, &run.sep))),
            ("linea EMOJI NUEVO", new_emoji_line(&run.search, &run.ds, &run.sep)),
            ("linea REGIONES", regions_line(&run.judge)),
            ("galeria", render_gallery(&run.search, &run.ds)),
            ("linea CONTROL NEGATIVO", control_line(&run.control)),
            ("lineas de aceptacion", render_acceptance(&run.checks)),
            ("linea RESULTADO EJ2", result_line(run)),
            ("dibujo del emoji nuevo junto al mas cercano", format_emojis_side_by_side(&[binarized, nearest])),
        ]
    })
}

/// Todos los numeros que el codigo (o un documento verificado contra el codigo) respalda.
fn corpus() -> &'static HashSet<String> {
    static CELL: OnceLock<HashSet<String>> = OnceLock::new();
    CELL.get_or_init(|| {
        let mut text = String::new();
        text += &ej1a().blob;
        text += &ej1b().blob;
        for (_, t) in ej2_texts() {
            text += t;
            text += "\n";
        }
        for doc in ["ej1a-analisis.md", "ej1b-analisis.md", "ej2-analisis.md"] {
            let path = format!("{}/docs/{doc}", env!("CARGO_MANIFEST_DIR"));
            text += &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
            text += "\n";
        }
        numeric_tokens(&text).into_iter().collect()
    })
}

// --------------------------------------------------------------------------------------------------
// Pruebas
// --------------------------------------------------------------------------------------------------

#[test]
fn numeric_token_rule_matches_the_python_verifier() {
    assert_eq!(
        numeric_tokens("dae_examples_p0.10.png 12 123 4.5 -0.25 1234abc x2 7 1e5 12345"),
        vec!["123", "4.5", "0.25", "12345"]
    );
}

#[test]
fn deck_is_a_stored_package() {
    let d = deck();
    assert!(d.all_stored);
    assert_eq!(d.member_names.first().map(String::as_str), Some("[Content_Types].xml"), "[Content_Types].xml va primero");
    assert!(d.slides.len() >= 20, "la presentacion tiene {} diapositivas y se esperaban al menos 20", d.slides.len());
    for (i, s) in d.slides.iter().enumerate() {
        assert_eq!(s.index, i + 1, "las partes de diapositiva deben numerarse sin huecos desde 1");
    }
}

#[test]
fn deck_has_a_section_per_exercise_in_order() {
    let d = deck();
    assert_eq!(d.slides[0].title, PORTADA, "la primera diapositiva es la portada");
    assert!(d.slides[1].title.starts_with("Resumen"), "la segunda diapositiva es el Resumen: {:?}", d.slides[1].title);
    let last = d.slides.last().unwrap();
    assert!(last.title.starts_with("Reproducibilidad"), "la ultima diapositiva {} es {:?}", last.index, last.title);
    let common = ["Arquitectura", "Técnicas", "Resultados", "Conclusiones"];
    let sections: [(&str, Vec<&str>); 3] = [
        ("Ej 1a", common.to_vec()),
        ("Ej 1b", common.to_vec()),
        ("Ej 2", ["Dataset"].into_iter().chain(common).collect()),
    ];
    let mut previous_last = 0;
    for (ej, tags) in &sections {
        let prefix = format!("{ej}: ");
        let indices: Vec<usize> = d.slides.iter().filter(|s| s.title.starts_with(&prefix)).map(|s| s.index).collect();
        assert!(!indices.is_empty(), "no hay diapositivas de {ej}");
        assert!(indices[0] > previous_last, "la seccion {ej} empieza en la diapositiva {} antes de que termine la anterior", indices[0]);
        previous_last = *indices.last().unwrap();
        for tag in tags {
            let full = format!("{prefix}{tag}");
            assert!(
                d.slides.iter().any(|s| s.title.starts_with(&full)),
                "falta una diapositiva titulada {full:?}..."
            );
        }
    }
    for s in &d.slides[2..d.slides.len() - 1] {
        assert!(
            ["Ej 1a: ", "Ej 1b: ", "Ej 2: "].iter().any(|p| s.title.starts_with(p)),
            "diapositiva {}: el titulo {:?} no pertenece a ninguna seccion de ejercicio",
            s.index,
            s.title
        );
    }
}

#[test]
fn deck_quotes_the_ej1a_lines_the_code_produces() {
    let f = ej1a();
    assert!(deck().has_run(&f.result_line), "falta el tramo exacto {:?}", f.result_line);
    assert!(deck().has_run(&f.point_line), "falta el tramo exacto {:?}", f.point_line);
    assert!(
        deck().has_run_containing(&f.config_name),
        "ningun tramo contiene el nombre de la configuracion elegida {:?}",
        f.config_name
    );
    assert!(f.sweep_rows.len() >= 18, "el barrido completo debe tener al menos 18 filas de variantes");
    for row in &f.sweep_rows {
        assert!(deck().has_row(row), "ninguna diapositiva tiene la fila completa y en orden {row:?} de comparison.md");
    }
}

#[test]
fn deck_quotes_the_ej1b_lines_the_code_produces() {
    let f = ej1b();
    assert_runs("tabla por nivel", &f.level_table);
    assert!(deck().has_run(&f.max_line), "falta el tramo exacto {:?}", f.max_line);
    assert!(deck().has_run(&f.selection), "falta el tramo exacto {:?}", f.selection);
    assert!(
        deck().has_run_containing(&f.config_name),
        "ningun tramo contiene el nombre de la configuracion elegida {:?}",
        f.config_name
    );
    assert!(f.comparison_rows.len() >= 13, "el barrido completo debe tener al menos 13 filas de variantes");
    for row in &f.comparison_rows {
        assert!(deck().has_row(row), "ninguna diapositiva tiene la fila completa y en orden {row:?} de dae_comparison.md");
    }
}

#[test]
fn deck_quotes_the_ej2_lines_the_code_produces() {
    for (label, text) in ej2_texts() {
        assert_runs(label, text);
    }
    assert_eq!(ej2_run().checks.len(), 11, "la corrida debe evaluar 11 verificaciones de aceptacion");
}

#[test]
fn deck_embeds_the_exercise_figures() {
    let pictures: HashSet<&str> =
        deck().slides.iter().flat_map(|s| s.pictures.iter().map(String::as_str)).collect();
    for path in [
        "output/ej2/emojis.png",
        "output/ej2/vae_loss.png",
        "output/ej2/vae_reconstruction.png",
        "output/ej2/vae_manifold.png",
        "output/ej2/vae_new_emoji.png",
        "output/ej2/plain_manifold.png",
        "output/ej1a/latent.png",
        "output/ej1a/new_letter.png",
        "output/ej1b/noise_examples.png",
        "output/ej1b/dae_error_vs_noise.png",
        "output/ej1b/dae_within_vs_noise.png",
    ] {
        assert!(pictures.contains(path), "ninguna diapositiva incrusta la imagen {path}");
    }
    let allowed: HashSet<String> =
        DEFAULT_LEVELS.iter().map(|&l| format!("output/ej1b/{}", dae_examples_filename(l))).collect();
    let mut examples = 0;
    for s in &deck().slides {
        for p in s.pictures.iter().filter(|p| p.contains("dae_examples_")) {
            examples += 1;
            assert!(allowed.contains(p), "diapositiva {}: la imagen {p:?} no es un archivo de ejemplos de un nivel por defecto", s.index);
        }
    }
    assert!(examples > 0, "la presentacion debe incrustar al menos una figura de ejemplos del Ej 1b");
}

#[test]
fn deck_numbers_come_from_the_code_or_verified_documents() {
    let corpus = corpus();
    for s in &deck().slides {
        for run in &s.runs {
            for token in numeric_tokens(run) {
                assert!(
                    corpus.contains(&token) || THEORY_CONSTANTS.contains(&token.as_str()),
                    "diapositiva {} ({:?}): el numero {token} no sale de lo que el codigo recalcula ni de los documentos de analisis; \
                     cite la linea del reporte o agregue su valor a la biblioteca",
                    s.index,
                    s.title
                );
            }
        }
    }
}

#[test]
fn deck_contains_no_retracted_claims() {
    for s in &deck().slides {
        let text = s.runs.join("\n").to_lowercase();
        for phrase in RETRACTED {
            assert!(!text.contains(phrase), "diapositiva {} ({:?}) contiene la afirmacion retirada {phrase:?}", s.index, s.title);
        }
    }
}
