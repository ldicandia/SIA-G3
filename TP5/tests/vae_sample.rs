//! Juicio de imagenes decodificadas contra el dataset, busqueda semillada de un emoji nuevo en el
//! prior y exportacion de la figura.

use std::path::PathBuf;
use std::sync::OnceLock;

use tp5::emoji::{build_emojis, separation, EmojiDataset, Separation};
use tp5::vae::{chosen_vae_config, train_vae, Vae, SAMPLE_SEED};
use tp5::vae_eval::{
    judge, new_emoji_line, prior_line, prior_stats, search_new_emoji, validate_draws, SampleSearch, DEFAULT_DRAWS,
    GALLERY_SIZE, PLAUSIBLE_FACTOR,
};
use tp5::vae_figures::{build_new_emoji_tiles, export_new_emoji, NEW_EMOJI_PNG};

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
        cfg.epochs = 1000;
        let (vae, _) = train_vae(&ds, &cfg).unwrap();
        (ds, sep, vae)
    })
}

/// Indices de pixeles en blanco en todos los emojis, tomados de los cuatro bloques de 3 x 3 de las
/// esquinas de la imagen de 24 x 24 (fuera del contorno de la cara).
fn corner_pixels(ds: &EmojiDataset) -> Vec<usize> {
    let mut out = Vec::new();
    for &r0 in &[0usize, 21] {
        for &c0 in &[0usize, 21] {
            for r in r0..r0 + 3 {
                for c in c0..c0 + 3 {
                    let k = r * 24 + c;
                    assert!(ds.x.column(k).iter().all(|&v| v < 0.5), "pixel {k} no esta en blanco");
                    out.push(k);
                }
            }
        }
    }
    out
}

/// El emoji 0 con `k` pixeles en blanco encendidos: esta a exactamente `k` px del emoji 0.
fn flipped(ds: &EmojiDataset, k: usize) -> ndarray::Array1<f64> {
    let mut p = ds.x.row(0).to_owned();
    for &i in corner_pixels(ds).iter().take(k) {
        p[i] = 1.0;
    }
    p
}

#[test]
fn judge_of_a_dataset_emoji_is_recognizable_and_not_new() {
    let (ds, sep, _) = model();
    for i in 0..ds.x.nrows() {
        let j = judge(ds, sep, ds.x.row(i));
        assert_eq!(j.nearest_index, i);
        assert_eq!(j.distance, 0);
        assert!(j.recognizable && j.plausible && !j.is_new);
        assert!(!j.new_and_plausible());
    }
}

#[test]
fn judge_thresholds_follow_the_dataset_separation() {
    let (ds, sep, _) = model();
    let near = sep.near_radius;
    let at = |k: usize| judge(ds, sep, flipped(ds, k).view());
    assert_eq!(at(sep.min - 1).distance, sep.min - 1);
    assert!(!at(sep.min - 1).is_new);
    assert!(at(sep.min).is_new);
    assert!(at(near).recognizable);
    assert!(!at(near + 1).recognizable);
    assert!(at(PLAUSIBLE_FACTOR * near).plausible);
    assert!(!at(PLAUSIBLE_FACTOR * near + 1).plausible);
    assert_eq!(at(near).nearest_index, 0);
}

#[test]
fn search_is_deterministic_by_seed_and_prefix_stable() {
    let (ds, _, vae) = model();
    let a = search_new_emoji(vae, ds, SAMPLE_SEED, 2000);
    let b = search_new_emoji(vae, ds, SAMPLE_SEED, 2000);
    assert_eq!(a.examined, b.examined);
    assert_eq!(a.accepted.len(), b.accepted.len());
    for (p, q) in a.accepted.iter().zip(&b.accepted) {
        assert_eq!(p.draw_index, q.draw_index);
        assert!(p.z.iter().zip(q.z.iter()).all(|(u, v)| u.to_bits() == v.to_bits()));
        assert!(p.image.iter().zip(q.image.iter()).all(|(u, v)| u.to_bits() == v.to_bits()));
        assert_eq!(p.judgement, q.judgement);
    }
    let c = search_new_emoji(vae, ds, SAMPLE_SEED ^ 1, 2000);
    assert!(!a.accepted.is_empty() && !c.accepted.is_empty());
    assert_ne!(a.accepted[0].z, c.accepted[0].z);

    let short = search_new_emoji(vae, ds, SAMPLE_SEED, 500);
    let long = search_new_emoji(vae, ds, SAMPLE_SEED, 2000);
    let first_long = long.accepted[0].draw_index;
    if first_long < 500 {
        assert_eq!(short.accepted[0].draw_index, first_long);
        assert_eq!(short.accepted[0].z, long.accepted[0].z);
    } else {
        assert!(short.accepted.is_empty());
    }
}

#[test]
fn accepted_samples_are_new_and_plausible_against_every_emoji() {
    let (ds, sep, vae) = model();
    let search = search_new_emoji(vae, ds, SAMPLE_SEED, 3000);
    assert!(!search.accepted.is_empty());
    assert!(search.accepted.len() <= GALLERY_SIZE);
    for s in &search.accepted {
        let mut best = (0usize, usize::MAX);
        for i in 0..ds.x.nrows() {
            let mut d = 0;
            for k in 0..576 {
                if (s.image[k] >= 0.5) != (ds.x[[i, k]] >= 0.5) {
                    d += 1;
                }
            }
            assert!(d >= sep.min, "emoji {i} a {d} px, minimo {}", sep.min);
            if d < best.1 {
                best = (i, d);
            }
        }
        assert_eq!(best.1, s.judgement.distance);
        assert_eq!(best.0, s.judgement.nearest_index);
        assert!(best.1 <= PLAUSIBLE_FACTOR * sep.near_radius);
    }
}

#[test]
fn search_finds_a_new_emoji_within_the_default_draws() {
    let (ds, _, vae) = model();
    let search = search_new_emoji(vae, ds, SAMPLE_SEED, DEFAULT_DRAWS);
    assert!(!search.accepted.is_empty(), "ningun emoji nuevo en {DEFAULT_DRAWS} sorteos");
}

#[test]
fn new_emoji_line_formats_success_and_failure() {
    let (ds, sep, vae) = model();
    let search = search_new_emoji(vae, ds, SAMPLE_SEED, DEFAULT_DRAWS);
    let line = new_emoji_line(&search, ds, sep);
    assert!(line.starts_with("EMOJI NUEVO: z=("), "{line}");
    assert!(line.contains(" sorteos | emoji mas cercano: "), "{line}");
    assert!(line.contains(&format!("a al menos {} px de todo el dataset", sep.min)), "{line}");
    let empty = SampleSearch { seed: 1, max_draws: 50, examined: 50, accepted: vec![] };
    let none = new_emoji_line(&empty, ds, sep);
    assert!(none.starts_with("EMOJI NUEVO: ninguno en 50 sorteos"), "{none}");
}

#[test]
fn prior_line_format_and_fractions_are_consistent() {
    let (ds, sep, vae) = model();
    let decode = |z: &ndarray::Array2<f64>| vae.decode(z);
    let s = prior_stats(&decode, 2, ds, sep, SAMPLE_SEED, 1000);
    assert_eq!(s.n, 1000);
    assert!(s.new_plausible_frac <= s.new_frac && s.new_frac <= 1.0);
    assert!(s.recognizable_frac <= 1.0 && s.recognizable_frac >= 0.0);
    assert!(prior_line(&s, sep).starts_with("PRIOR: de 1000 puntos"));
    let again = prior_stats(&decode, 2, ds, sep, SAMPLE_SEED, 1000);
    assert_eq!(s, again);
}

#[test]
fn new_emoji_png_is_written_with_the_fixed_name_and_fails_when_empty() {
    let (ds, _, vae) = model();
    let search = search_new_emoji(vae, ds, SAMPLE_SEED, DEFAULT_DRAWS);
    let k = search.accepted.len();
    let tiles = build_new_emoji_tiles(&search, ds);
    assert_eq!(tiles.len(), 2 * k);
    assert!(tiles[0].0.starts_with("nuevo 1: "));
    assert_eq!(tiles[k].0, ds.names[search.accepted[0].judgement.nearest_index]);

    let dir = unique_tmp_dir("new-emoji");
    let path = export_new_emoji(&search, ds, &dir).unwrap();
    assert_eq!(path, dir.join(NEW_EMOJI_PNG));
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..8], &PNG_SIGNATURE);
    assert!(bytes.len() > 3000);

    let empty = SampleSearch { seed: 1, max_draws: 10, examined: 10, accepted: vec![] };
    let dir2 = unique_tmp_dir("new-emoji-empty");
    let err = export_new_emoji(&empty, ds, &dir2).unwrap_err().to_string();
    assert!(err.contains("10 sorteos"), "{err}");
    assert!(!dir2.join(NEW_EMOJI_PNG).exists());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn validate_draws_bounds() {
    assert!(validate_draws(0).is_err());
    assert!(validate_draws(100_001).is_err());
    assert_eq!(validate_draws(1).unwrap(), 1);
    assert_eq!(validate_draws(100_000).unwrap(), 100_000);
}
