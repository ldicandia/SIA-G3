//! Lectura de `docs/font.h`: 32 patrones de 7x5 pixeles como matriz 32 x 35 de 0.0 / 1.0.

use anyhow::{anyhow, bail, Context, Result};
use ndarray::{Array2, ArrayView1};

pub const ROWS: usize = 7;
pub const COLS: usize = 5;
pub const N_PIXELS: usize = ROWS * COLS;
pub const N_PATTERNS: usize = 32;

/// Dataset de patrones binarios: `x` tiene forma 32 x 35 (fila a fila, columna 0 = izquierda).
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Codigo del caracter (comentario final de cada linea de `font.h`).
    pub codes: Vec<u8>,
    /// Etiqueta legible: el caracter ASCII, o "DEL" para 0x7f.
    pub labels: Vec<String>,
    pub x: Array2<f64>,
}

fn parse_hex_token(tok: &str) -> Result<u32> {
    let t = tok.trim();
    let digits = t
        .strip_prefix("0x")
        .or_else(|| t.strip_prefix("0X"))
        .ok_or_else(|| anyhow!("token hexadecimal invalido: {t:?}"))?;
    u32::from_str_radix(digits, 16).with_context(|| format!("token hexadecimal invalido: {t:?}"))
}

fn label_for(code: u8) -> String {
    if code == 0x7f {
        "DEL".to_string()
    } else {
        (code as char).to_string()
    }
}

/// Parsea el texto de `font.h`. Nunca entra en panico: ante entrada mal formada devuelve `Err`.
pub fn parse_font_h(src: &str) -> Result<Dataset> {
    let mut codes = Vec::new();
    let mut labels = Vec::new();
    let mut rows: Vec<[f64; N_PIXELS]> = Vec::new();

    for (lineno, line) in src.lines().enumerate() {
        // El comentario puede contener llaves (`{`, `}`), por eso se separa primero.
        let (code_part, comment) = match line.split_once("//") {
            Some((c, m)) => (c, Some(m)),
            None => (line, None),
        };
        let (Some(open), Some(close)) = (code_part.find('{'), code_part.rfind('}')) else {
            continue;
        };
        if close < open {
            continue;
        }
        let inner = &code_part[open + 1..close];
        let tokens: Vec<&str> = inner.split(',').map(str::trim).filter(|t| !t.is_empty()).collect();
        if tokens.len() != ROWS {
            bail!("linea {}: se esperaban {ROWS} bytes y hay {}", lineno + 1, tokens.len());
        }
        let mut pixels = [0.0_f64; N_PIXELS];
        for (r, tok) in tokens.iter().enumerate() {
            let byte = parse_hex_token(tok).with_context(|| format!("linea {}", lineno + 1))?;
            if byte > 0x1f {
                bail!("linea {}: byte 0x{byte:02x} fuera de rango (maximo 0x1f)", lineno + 1);
            }
            for c in 0..COLS {
                if (byte >> (COLS - 1 - c)) & 1 == 1 {
                    pixels[r * COLS + c] = 1.0;
                }
            }
        }
        let comment = comment.ok_or_else(|| anyhow!("linea {}: falta el comentario con el codigo", lineno + 1))?;
        let code_tok = comment
            .split_whitespace()
            .next()
            .ok_or_else(|| anyhow!("linea {}: comentario vacio", lineno + 1))?
            .trim_end_matches(',');
        let code = parse_hex_token(code_tok).with_context(|| format!("linea {}: codigo de caracter", lineno + 1))?;
        let code = u8::try_from(code).map_err(|_| anyhow!("linea {}: codigo de caracter fuera de rango", lineno + 1))?;
        codes.push(code);
        labels.push(label_for(code));
        rows.push(pixels);
    }

    if rows.len() != N_PATTERNS {
        bail!("se esperaban {N_PATTERNS} patrones y se encontraron {}", rows.len());
    }
    let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    let x = Array2::from_shape_vec((N_PATTERNS, N_PIXELS), flat)?;
    Ok(Dataset { codes, labels, x })
}

/// Carga el `font.h` embebido en el binario.
pub fn load_font() -> Result<Dataset> {
    parse_font_h(include_str!("../docs/font.h"))
}

/// Dibuja un patron de 35 valores como 7 lineas de 5 caracteres (`#` encendido, `.` apagado).
pub fn format_grid(pattern: ArrayView1<f64>) -> String {
    (0..ROWS)
        .map(|r| {
            (0..COLS)
                .map(|c| if pattern[r * COLS + c] >= 0.5 { '#' } else { '.' })
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn font_src() -> &'static str {
        include_str!("../docs/font.h")
    }

    fn grid_of(label: &str) -> Vec<String> {
        let ds = load_font().unwrap();
        let i = ds.labels.iter().position(|l| l == label).unwrap();
        format_grid(ds.x.row(i)).split('\n').map(str::to_string).collect()
    }

    #[test]
    fn parses_32_patterns_with_shape_32_by_35() {
        let ds = load_font().unwrap();
        assert_eq!(ds.x.dim(), (32, 35));
        assert!(ds.x.iter().all(|&v| v == 0.0 || v == 1.0));
        assert_eq!(ds.codes.len(), 32);
    }

    #[test]
    fn labels_run_from_backtick_to_del() {
        let ds = load_font().unwrap();
        assert_eq!(ds.labels[0], "`");
        assert_eq!(ds.labels[31], "DEL");
        assert_eq!(ds.codes[0], 0x60);
        assert_eq!(ds.codes[31], 0x7f);
        let mut sorted = ds.labels.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 32);
    }

    #[test]
    fn grid_of_l_matches_font() {
        let mut expected = vec![".##..".to_string()];
        expected.extend(std::iter::repeat("..#..".to_string()).take(6));
        assert_eq!(grid_of("l"), expected);
    }

    #[test]
    fn grid_of_b_matches_font() {
        assert_eq!(
            grid_of("b"),
            vec!["#....", "#....", "#....", "###..", "#..#.", "#..#.", "###.."]
        );
    }

    #[test]
    fn grid_of_del_is_all_ones() {
        assert_eq!(grid_of("DEL"), vec!["#####"; 7]);
    }

    #[test]
    fn malformed_input_returns_err() {
        // 31 patrones: se elimina la linea del backtick.
        let missing: String = font_src()
            .lines()
            .filter(|l| !l.contains("0x60"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(parse_font_h(&missing).is_err());
        // Byte fuera de rango (0x20 > 0x1f).
        let big = font_src().replacen("0x04, 0x04, 0x02", "0x20, 0x04, 0x02", 1);
        assert!(parse_font_h(&big).is_err());
        // Token no hexadecimal.
        let bad = font_src().replacen("0x04, 0x04, 0x02", "0xzz, 0x04, 0x02", 1);
        assert!(parse_font_h(&bad).is_err());
    }
}
