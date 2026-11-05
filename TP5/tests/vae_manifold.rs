//! Grilla del plano latente decodificada: orientacion, validacion, metricas y figura.

use std::path::PathBuf;
use std::sync::OnceLock;

use ndarray::{array, Array1, Array2};

use tp5::emoji::{build_emojis, separation, EmojiDataset, Separation};
use tp5::vae::{chosen_vae_config, train_vae, Vae};
use tp5::vae_eval::{
    decode_grid, latent_grid, manifold_line, manifold_metrics, validate_grid, GRID_HALF_RANGE, GRID_SIDE_DEFAULT,
    LARGE_STEP_FACTOR,
};
use tp5::vae_figures::{build_manifold_tiles, export_manifold, MANIFOLD_PNG};

const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

fn unique_tmp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tp5-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn model() -> &'static (EmojiDataset, Separation, Vae) {
    static MODEL: OnceLock<(EmojiDataset, Separation, Vae)> = OnceLock::new();
    MODEL.get_or_init(|| {
        let ds = build_emojis();
        let sep = separation(&ds);
        let mut cfg = chosen_vae_config();
        cfg.epochs = 300;
        let (vae, _) = train_vae(&ds, &cfg).unwrap();
        (ds, sep, vae)
    })
}

fn wrong(a: &Array1<f64>, b: &Array1<f64>) -> usize {
    (0..a.len()).filter(|&k| (a[k] >= 0.5) != (b[k] >= 0.5)).count()
}

#[test]
fn latent_grid_has_side_squared_points_and_the_stated_orientation() {
    let g = latent_grid(3, 2.5);
    assert_eq!(g.dim(), (9, 2));
    assert_eq!(g.row(0).to_vec(), vec![-2.5, 2.5]);
    assert_eq!(g.row(1).to_vec(), vec![0.0, 2.5]);
    assert_eq!(g.row(2).to_vec(), vec![2.5, 2.5]);
    for c in 0..3 {
        assert_eq!(g[[6 + c, 1]], -2.5, "la ultima fila de celdas tiene z2 = -2.5");
    }
    // Fila mayor: la celda (r, c) esta en la posicion r * side + c.
    assert_eq!(g.row(3).to_vec(), vec![-2.5, 0.0]);
    assert_eq!(g.row(5).to_vec(), vec![2.5, 0.0]);
    assert_eq!(latent_grid(15, GRID_HALF_RANGE).dim(), (225, 2));
}

#[test]
fn validate_grid_bounds() {
    assert!(validate_grid(2).is_err());
    assert!(validate_grid(42).is_err());
    assert_eq!(validate_grid(3).unwrap(), 3);
    assert_eq!(validate_grid(41).unwrap(), 41);
    let msg = validate_grid(42).unwrap_err().to_string();
    assert!(msg.contains("42"), "{msg}");
}

#[test]
fn decode_grid_row_equals_decoding_that_point_alone() {
    let (_, _, vae) = model();
    let grid = decode_grid(vae, 5, GRID_HALF_RANGE);
    assert_eq!(grid.dim(), (25, 576));
    let g = latent_grid(5, GRID_HALF_RANGE);
    let cell = 5 + 2;
    let single = vae.decode(&g.slice(ndarray::s![cell..cell + 1, ..]).to_owned());
    for k in 0..576 {
        assert!((grid[[cell, k]] - single[[0, k]]).abs() < 1e-12);
    }
}

#[test]
fn manifold_metrics_match_an_independent_computation() {
    let (ds, sep, _) = model();
    let decoded: Array2<f64> = ds.x.slice(ndarray::s![0..9, ..]).to_owned();
    let m = manifold_metrics(&decoded, 3, ds, sep);
    assert_eq!(m.recognizable_frac, 1.0);
    assert_eq!(m.n_cells, 9);
    let mut steps = Vec::new();
    for r in 0..3 {
        for c in 0..3 {
            let here = decoded.row(r * 3 + c).to_owned();
            if c + 1 < 3 {
                steps.push(wrong(&here, &decoded.row(r * 3 + c + 1).to_owned()));
            }
            if r + 1 < 3 {
                steps.push(wrong(&here, &decoded.row((r + 1) * 3 + c).to_owned()));
            }
        }
    }
    assert_eq!(steps.len(), 12);
    let mean = steps.iter().sum::<usize>() as f64 / 12.0;
    let large = steps.iter().filter(|&&s| s > LARGE_STEP_FACTOR * sep.near_radius).count() as f64 / 12.0;
    assert!((m.mean_adjacent_step - mean).abs() < 1e-12);
    assert_eq!(m.max_adjacent_step, *steps.iter().max().unwrap());
    assert!((m.large_step_frac - large).abs() < 1e-12);
}

#[test]
fn manifold_metrics_count_large_steps() {
    let (ds, sep, _) = model();
    let big = LARGE_STEP_FACTOR * sep.near_radius + 6;
    let base = ds.x.row(0).to_owned();
    let mut loud = base.clone();
    for k in 0..big {
        assert!(base[k] < 0.5, "el pixel {k} del emoji 0 deberia estar en blanco");
        loud[k] = 1.0;
    }
    let mut decoded = Array2::zeros((4, 576));
    for (i, row) in [&base, &loud, &base, &base].iter().enumerate() {
        decoded.row_mut(i).assign(row);
    }
    let m = manifold_metrics(&decoded, 2, ds, sep);
    assert_eq!(m.max_adjacent_step, big);
    // Pares: (0,1) = big, (0,2) = 0, (1,3) = big, (2,3) = 0.
    assert_eq!(m.large_step_frac, 0.5);
    assert!((m.mean_adjacent_step - big as f64 / 2.0).abs() < 1e-12);
    assert!(m.recognizable_frac < 1.0 && m.recognizable_frac >= 0.5);
}

#[test]
fn manifold_line_format() {
    let (ds, sep, vae) = model();
    let decoded = decode_grid(vae, GRID_SIDE_DEFAULT, GRID_HALF_RANGE);
    let m = manifold_metrics(&decoded, GRID_SIDE_DEFAULT, ds, sep);
    let line = manifold_line(&m, sep);
    assert!(line.starts_with("MANIFOLD: 225 celdas (15x15)"), "{line}");
    assert!(line.contains("reconocibles "), "{line}");
    assert!(line.contains("salto medio entre celdas vecinas "), "{line}");
}

#[test]
fn manifold_tiles_have_one_empty_caption_per_cell() {
    let decoded = array![[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]];
    let tiles = build_manifold_tiles(&decoded);
    assert_eq!(tiles.len(), 3);
    assert!(tiles.iter().all(|(c, _)| c.is_empty()));
    assert_eq!(tiles[1].1, array![1.0, 0.0]);
}

#[test]
fn manifold_png_is_written_with_the_fixed_name_and_rejects_a_mismatched_size() {
    let (_, _, vae) = model();
    let decoded = decode_grid(vae, 5, GRID_HALF_RANGE);
    let dir = unique_tmp_dir("manifold");
    let path = export_manifold(&decoded, 5, &dir).unwrap();
    assert_eq!(path, dir.join(MANIFOLD_PNG));
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..8], &PNG_SIGNATURE);
    assert!(bytes.len() > 3000);
    let dir2 = unique_tmp_dir("manifold-bad");
    let err = export_manifold(&decoded, 6, &dir2).unwrap_err().to_string();
    assert!(err.contains("36"), "{err}");
    assert!(!dir2.join(MANIFOLD_PNG).exists());
    let _ = std::fs::remove_dir_all(&dir);
}
