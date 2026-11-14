//! Escritura de PNG con plotters y la guarda de legibilidad de las figuras.

use ndarray::Array1;
use tp5::plots::{
    check_legible, effective_pt, min_font_px, plot_gray_tiles, plot_pattern_rows, plot_pattern_rows_in_blocks,
    plot_series, PatternRow, Series, MIN_FIGURE_TEXT_PT, SLIDE_CONTENT_H_IN, SLIDE_CONTENT_W_IN,
};

fn unique_tmp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("tp5-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn series(name: &str, f: impl Fn(f64) -> f64) -> Series {
    Series { name: name.to_string(), points: (1..=50).map(|i| (i as f64, f(i as f64))).collect() }
}

fn png_size(path: &std::path::Path) -> (u32, u32) {
    let b = std::fs::read(path).unwrap();
    assert_eq!(&b[..8], &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
    (u32::from_be_bytes([b[16], b[17], b[18], b[19]]), u32::from_be_bytes([b[20], b[21], b[22], b[23]]))
}

#[test]
fn plot_series_writes_png() {
    let dir = unique_tmp_dir("plots");
    let path = dir.join("a.png");
    let s = [series("uno", |x| 1.0 / x), series("dos", |x| 2.0 / (x + 1.0))];
    plot_series(&path, "titulo", "epoca", "perdida", &s, false).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..8], &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
    assert!(bytes.len() > 1000);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn plot_series_rejects_empty_input() {
    let dir = unique_tmp_dir("plots-empty");
    assert!(plot_series(&dir.join("e.png"), "t", "x", "y", &[], false).is_err());
    let vacia = [Series { name: "v".into(), points: vec![] }];
    assert!(plot_series(&dir.join("e.png"), "t", "x", "y", &vacia, false).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn log_axis_tolerates_zero_values() {
    let dir = unique_tmp_dir("plots-log");
    let s = [series("con cero", |x| if x < 10.0 { 0.0 } else { 1.0 / x })];
    plot_series(&dir.join("l.png"), "t", "x", "y", &s, true).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn effective_pt_is_the_font_at_the_alone_on_slide_scale() {
    // 1000 x 600: limita el alto, 5.35 / 600 pulgadas por pixel.
    assert!((effective_pt(26, 1000, 600) - 26.0 * 5.35 / 600.0 * 72.0).abs() < 1e-9);
    // 2000 x 400: limita el ancho, 12.333 / 2000 pulgadas por pixel.
    assert!((effective_pt(40, 2000, 400) - 40.0 * 12.333 / 2000.0 * 72.0).abs() < 1e-9);
    assert_eq!(min_font_px(1000, 600), 25);
    assert!(effective_pt(min_font_px(1000, 600), 1000, 600) >= MIN_FIGURE_TEXT_PT);
    assert!(effective_pt(min_font_px(1000, 600) - 1, 1000, 600) < MIN_FIGURE_TEXT_PT);
}

#[test]
fn the_guard_rejects_an_illegible_combination() {
    assert!(check_legible("prueba", 12, 1000, 600).is_err(), "12 px en 1000 x 600 son 7.7 pt");
    assert!(check_legible("prueba", 26, 1000, 600).is_ok());
    let msg = check_legible("prueba", 12, 1000, 600).unwrap_err().to_string();
    assert!(msg.contains("16"), "{msg}");
}

#[test]
fn plot_series_refuses_a_legend_that_cannot_be_legible() {
    let dir = unique_tmp_dir("plots-legend");
    let many: Vec<Series> = (0..12).map(|i| series(&format!("serie {i}"), |x| x)).collect();
    assert!(plot_series(&dir.join("m.png"), "t", "x", "y", &many, false).is_err(), "12 series no entran");
    let long = [series(&"nombre muy largo ".repeat(8), |x| x)];
    assert!(plot_series(&dir.join("n.png"), "t", "x", "y", &long, false).is_err(), "nombre sin lugar");
    assert!(!dir.join("m.png").exists() && !dir.join("n.png").exists());
}

fn letter_rows(n_rows: usize, caption: bool) -> Vec<PatternRow> {
    (0..n_rows)
        .map(|r| PatternRow {
            label: format!("0.{r}0"),
            cells: (0..8)
                .map(|i| {
                    let cap = if caption { format!("{i} px") } else { String::new() };
                    (cap, Array1::from_shape_fn(35, |p| ((p + i + r) % 2) as f64))
                })
                .collect(),
        })
        .collect()
}

#[test]
fn pattern_rows_return_an_error_instead_of_an_illegible_figure() {
    let dir = unique_tmp_dir("plots-rows");
    // 24 filas con pie por celda: cada fila agrega mas alto del que la letra legible puede cubrir.
    assert!(plot_pattern_rows(&dir.join("a.png"), "t", &letter_rows(24, true)).is_err());
    // Una etiqueta de fila que no deja lugar a las letras tampoco se achica.
    let mut wide = letter_rows(3, false);
    wide[0].label = "etiqueta larga ".repeat(12);
    assert!(plot_pattern_rows_in_blocks(&dir.join("b.png"), "t", &wide, 2).is_err());
    assert!(!dir.join("a.png").exists() && !dir.join("b.png").exists());
    // Las formas reales: 9 filas sin pies en dos bloques, y 3 filas con pies.
    for (name, rows, blocks) in [("c.png", letter_rows(9, false), 2), ("d.png", letter_rows(3, true), 1)] {
        let path = dir.join(name);
        plot_pattern_rows_in_blocks(&path, "titulo", &rows, blocks).unwrap();
        let (w, h) = png_size(&path);
        assert!(min_font_px(w, h) as f64 <= 7.0 * 28.0, "{name}: {w} x {h}");
    }
    let (w, h) = png_size(&dir.join("c.png"));
    assert!(w as f64 / h as f64 > 2.0, "dos bloques dan una figura apaisada: {w} x {h}");
    assert!(plot_pattern_rows_in_blocks(&dir.join("e.png"), "t", &letter_rows(3, false), 0).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn gray_tiles_return_an_error_instead_of_an_illegible_figure() {
    let dir = unique_tmp_dir("plots-tiles");
    let tile = |c: &str| (c.to_string(), Array1::from_elem(24 * 24, 0.5));
    // 36 nombres en una sola fila: el pie no llega a 16 pt por mas que se separen los tiles.
    let named: Vec<_> = (0..36).map(|i| tile(&format!("patron-{i}"))).collect();
    assert!(plot_gray_tiles(&dir.join("a.png"), "t", &named, 24, 36, 4).is_err());
    assert!(!dir.join("a.png").exists());
    // Formas tipicas de mosaico: 36 sin pies, 72 con pie en filas alternas, 15 x 15 sin pies, tira de 16.
    let bare: Vec<_> = (0..36).map(|_| tile("")).collect();
    plot_gray_tiles(&dir.join("b.png"), "titulo", &bare, 24, 6, 4).unwrap();
    let recon: Vec<_> = (0..72).map(|i| if (i / 12) % 2 == 1 { tile("16 px") } else { tile("") }).collect();
    plot_gray_tiles(&dir.join("c.png"), "titulo", &recon, 24, 12, 3).unwrap();
    let grid: Vec<_> = (0..225).map(|_| tile("")).collect();
    plot_gray_tiles(&dir.join("d.png"), "titulo", &grid, 24, 15, 2).unwrap();
    let strip: Vec<_> = (0..16).map(|i| if i < 8 { tile("1: 25 px") } else { tile("patron-x") }).collect();
    plot_gray_tiles(&dir.join("e.png"), "titulo", &strip, 24, 8, 4).unwrap();
    for name in ["b.png", "c.png", "d.png", "e.png"] {
        let (w, h) = png_size(&dir.join(name));
        assert!(w > 0 && h > 0, "{name}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Las constantes del area de contenido son las de presentacion/pptx_minimo.py.
#[test]
fn slide_content_area_matches_the_pptx_writer() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/presentacion/pptx_minimo.py");
    let text = std::fs::read_to_string(path).unwrap();
    let value = |name: &str| -> f64 {
        let line = text
            .lines()
            .find(|l| l.starts_with(&format!("{name} = ")))
            .unwrap_or_else(|| panic!("{path}: falta la constante {name}"));
        line[name.len() + 3..].trim().parse().unwrap_or_else(|e| panic!("{name}: {e}"))
    };
    let (ancho, alto, margen, contenido_y) =
        (value("ANCHO_IN"), value("ALTO_IN"), value("MARGEN_IN"), value("CONTENIDO_Y_IN"));
    assert!((ancho - 2.0 * margen - SLIDE_CONTENT_W_IN).abs() < 1e-9, "ancho de contenido");
    assert!((alto - margen - contenido_y - SLIDE_CONTENT_H_IN).abs() < 1e-9, "alto de contenido");
}
