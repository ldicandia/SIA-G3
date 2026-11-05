//! Juicio por region de las imagenes decodificadas, busqueda semillada de emojis nuevos en el prior
//! (deduplicada y ordenada por margen), galeria y exportacion de la figura.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::OnceLock;

use tp5::emoji::{build_emojis, separation, style_regions, EmojiDataset, Separation, EMOJI_EYES, EMOJI_MOUTHS};
use tp5::vae::{chosen_vae_config, train_vae, Vae, SAMPLE_SEED};
use tp5::vae_eval::{
    judge, new_emoji_line, prior_line, prior_stats, render_gallery, search_new_emoji, validate_draws, NoiseFloors,
    SampleSearch, StyleJudge, BLEND_SLACK, DEFAULT_DRAWS, GALLERY_SIZE,
};
use tp5::vae_figures::{build_new_emoji_tiles, export_new_emoji, NEW_EMOJI_PNG};

const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

fn unique_tmp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tp5-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// El VAE de la configuracion congelada, entrenado una sola vez para todo el archivo.
fn model() -> &'static (EmojiDataset, Separation, Vae) {
    static MODEL: OnceLock<(EmojiDataset, Separation, Vae)> = OnceLock::new();
    MODEL.get_or_init(|| {
        let ds = build_emojis();
        let sep = separation(&ds);
        let (vae, _) = train_vae(&ds, &chosen_vae_config()).unwrap();
        (ds, sep, vae)
    })
}

fn empty_search(max_draws: usize) -> SampleSearch {
    SampleSearch {
        seed: 1,
        max_draws,
        examined: max_draws,
        floors: NoiseFloors { eyes: 0, mouth: 0, frame: 0 },
        candidates: 0,
        distinct_keys: 0,
        accepted: vec![],
    }
}

fn key_of(s: &tp5::vae_eval::NewEmoji) -> (usize, usize, usize) {
    (s.judgement.nearest_index, s.verdict.eyes.style, s.verdict.mouth.style)
}

fn best_margin(search: &SampleSearch) -> usize {
    search.accepted.iter().map(|s| s.verdict.margin).max().unwrap_or(0)
}

#[test]
fn judge_of_a_dataset_emoji_is_recognizable() {
    let (ds, sep, _) = model();
    for i in 0..ds.x.nrows() {
        let j = judge(ds, sep, ds.x.row(i));
        assert_eq!(j.nearest_index, i);
        assert_eq!(j.distance, 0);
        assert!(j.recognizable);
    }
}

#[test]
fn search_is_deterministic_by_seed() {
    let (ds, _, vae) = model();
    let a = search_new_emoji(vae, ds, SAMPLE_SEED, 2000);
    let b = search_new_emoji(vae, ds, SAMPLE_SEED, 2000);
    assert_eq!(a.examined, b.examined);
    assert_eq!(a.candidates, b.candidates);
    assert_eq!(a.distinct_keys, b.distinct_keys);
    assert_eq!(a.floors, b.floors);
    assert_eq!(a.accepted.len(), b.accepted.len());
    for (p, q) in a.accepted.iter().zip(&b.accepted) {
        assert_eq!(p.draw_index, q.draw_index);
        assert!(p.z.iter().zip(q.z.iter()).all(|(u, v)| u.to_bits() == v.to_bits()));
        assert!(p.image.iter().zip(q.image.iter()).all(|(u, v)| u.to_bits() == v.to_bits()));
        assert_eq!(p.judgement, q.judgement);
        assert_eq!(p.verdict, q.verdict);
    }
    let c = search_new_emoji(vae, ds, SAMPLE_SEED ^ 1, 2000);
    assert!(!a.accepted.is_empty() && !c.accepted.is_empty());
    assert_ne!(a.accepted[0].z, c.accepted[0].z);
}

#[test]
fn longer_search_never_lowers_the_best_margin() {
    let (ds, _, vae) = model();
    let short = search_new_emoji(vae, ds, SAMPLE_SEED, 500);
    let long = search_new_emoji(vae, ds, SAMPLE_SEED, 2000);
    assert!(long.examined > short.examined);
    assert!(long.candidates >= short.candidates);
    assert!(best_margin(&long) >= best_margin(&short));
}

#[test]
fn gallery_is_deduplicated_and_ranked_by_margin() {
    let (ds, _, vae) = model();
    let search = search_new_emoji(vae, ds, SAMPLE_SEED, DEFAULT_DRAWS);
    assert!(!search.accepted.is_empty());
    assert!(search.accepted.len() <= GALLERY_SIZE);
    let keys: HashSet<(usize, usize, usize)> = search.accepted.iter().map(key_of).collect();
    assert_eq!(keys.len(), search.accepted.len(), "hay combinaciones repetidas en la galeria");
    for w in search.accepted.windows(2) {
        let (p, q) = (&w[0].verdict.margin, &w[1].verdict.margin);
        assert!(p >= q, "margenes no decrecientes: {p} luego {q}");
        if p == q {
            assert!(w[0].draw_index < w[1].draw_index, "empates ordenados por sorteo");
        }
    }
    assert!(search.distinct_keys >= search.accepted.len());
    assert!(search.candidates >= search.distinct_keys);
}

#[test]
fn accepted_samples_are_hybrids_verified_independently() {
    let (ds, _, vae) = model();
    let regions = style_regions(ds);
    // pisos recalculados con bucles simples sobre las reconstrucciones del modelo
    let recon = vae.reconstruct_mean(&ds.x);
    let worst = |pixels: &[usize]| {
        let mut w = 0usize;
        for i in 0..36 {
            let mut d = 0usize;
            for &k in pixels {
                if (recon[[i, k]] >= 0.5) != (ds.x[[i, k]] >= 0.5) {
                    d += 1;
                }
            }
            w = w.max(d);
        }
        w
    };
    let floors = NoiseFloors { eyes: worst(&regions.eyes), mouth: worst(&regions.mouth), frame: worst(&regions.frame) };

    let search = search_new_emoji(vae, ds, SAMPLE_SEED, DEFAULT_DRAWS);
    assert_eq!(search.floors, floors);
    assert!(!search.accepted.is_empty());
    // el estilo de ojos e es la fila 6 * e del dataset y el de boca m es la fila m
    let eyes_proto = |e: usize| e * 6;
    let mouth_proto = |m: usize| m;
    for s in &search.accepted {
        let bin = |k: usize| s.image[k] >= 0.5;
        let dists = |pixels: &[usize], row_of: &dyn Fn(usize) -> usize| -> Vec<usize> {
            (0..6)
                .map(|st| pixels.iter().filter(|&&k| bin(k) != (ds.x[[row_of(st), k]] >= 0.5)).count())
                .collect()
        };
        let gap = |pixels: &[usize], row_of: &dyn Fn(usize) -> usize, a: usize, b: usize| -> usize {
            pixels.iter().filter(|&&k| (ds.x[[row_of(a), k]] >= 0.5) != (ds.x[[row_of(b), k]] >= 0.5)).count()
        };
        let (de, dm) = (dists(&regions.eyes, &eyes_proto), dists(&regions.mouth, &mouth_proto));
        let (min_e, arg_e) = de.iter().enumerate().map(|(i, &d)| (d, i)).min().unwrap();
        let (min_m, arg_m) = dm.iter().enumerate().map(|(i, &d)| (d, i)).min().unwrap();
        assert!(min_e > floors.eyes || min_m > floors.mouth, "ninguna region supera su piso");
        assert_eq!((s.verdict.eyes.distance, s.verdict.eyes.style), (min_e, arg_e));
        assert_eq!((s.verdict.mouth.distance, s.verdict.mouth.style), (min_m, arg_m));

        let frame_distance = regions.frame.iter().filter(|&&k| bin(k) != (ds.x[[0, k]] >= 0.5)).count();
        assert!(frame_distance <= floors.frame, "marco roto: {frame_distance} px");
        assert_eq!(s.verdict.frame_distance, frame_distance);

        let mut margin = 0usize;
        for (pixels, row_of, d, min, floor, v) in [
            (&regions.eyes, &eyes_proto as &dyn Fn(usize) -> usize, &de, min_e, floors.eyes, &s.verdict.eyes),
            (&regions.mouth, &mouth_proto as &dyn Fn(usize) -> usize, &dm, min_m, floors.mouth, &s.verdict.mouth),
        ] {
            assert_eq!(v.off_style, min > floor);
            if v.off_style {
                // segundo mas cercano y regla de mezcla recalculados
                let mut order: Vec<usize> = (0..6).collect();
                order.sort_by_key(|&st| (d[st], st));
                assert_eq!((v.second_style, v.second_distance), (order[1], d[order[1]]));
                let between = (d[order[0]] + d[order[1]]) as f64
                    <= BLEND_SLACK * gap(pixels, row_of, order[0], order[1]) as f64;
                assert!(between, "la region fuera de estilo no queda entre sus dos estilos mas cercanos");
                assert!(v.blend_ok);
                margin = margin.max(min - floor);
            }
        }
        assert_eq!(s.verdict.margin, margin);
        assert!(s.verdict.plausible && s.verdict.hybrid);
    }
}

#[test]
fn search_finds_a_full_gallery_within_the_default_draws() {
    let (ds, _, vae) = model();
    let search = search_new_emoji(vae, ds, SAMPLE_SEED, DEFAULT_DRAWS);
    assert_eq!(search.accepted.len(), GALLERY_SIZE, "solo {} combinaciones distintas", search.distinct_keys);
}

#[test]
fn new_emoji_line_formats_success_and_failure() {
    let (ds, sep, vae) = model();
    let search = search_new_emoji(vae, ds, SAMPLE_SEED, DEFAULT_DRAWS);
    let line = new_emoji_line(&search, ds, sep);
    assert!(line.starts_with("EMOJI NUEVO: z=("), "{line}");
    assert!(line.contains(" sorteo "), "{line}");
    assert!(line.contains("emoji mas cercano: "), "{line}");
    assert!(line.contains("del estilo mas cercano (piso "), "{line}");
    assert!(line.contains("radio de cercania "), "{line}");
    let h = &search.accepted[0];
    assert!(line.contains(&format!("ojos: {} a {} px", EMOJI_EYES[h.verdict.eyes.style], h.verdict.eyes.distance)));
    assert!(line.contains(&format!("boca: {} a {} px", EMOJI_MOUTHS[h.verdict.mouth.style], h.verdict.mouth.distance)));
    let none = new_emoji_line(&empty_search(50), ds, sep);
    assert!(none.starts_with("EMOJI NUEVO: ninguno en 50 sorteos"), "{none}");
}

#[test]
fn gallery_lines_have_the_exact_format() {
    let (ds, _, vae) = model();
    let search = search_new_emoji(vae, ds, SAMPLE_SEED, DEFAULT_DRAWS);
    let text = render_gallery(&search, ds);
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 1 + search.accepted.len());
    assert!(lines[0].starts_with("GALERIA: "), "{}", lines[0]);
    for (i, l) in lines.iter().enumerate().skip(1) {
        assert!(l.starts_with(&format!("  nuevo {i}: sorteo ")), "{l}");
        assert!(l.contains(" | margen "), "{l}");
    }
}

#[test]
fn prior_stats_are_consistent_and_deterministic() {
    let (ds, _, vae) = model();
    let judge = StyleJudge::new(ds, &vae.reconstruct_mean(&ds.x));
    let decode = |z: &ndarray::Array2<f64>| vae.decode(z);
    let s = prior_stats(&decode, 2, &judge, SAMPLE_SEED, 1000);
    assert_eq!(s.n, 1000);
    assert!(s.hybrid_frac <= s.plausible_frac, "{s:?}");
    assert!(s.hybrid_frac <= s.off_style_frac, "{s:?}");
    for f in [s.recognizable_frac, s.plausible_frac, s.off_style_frac, s.hybrid_frac] {
        assert!((0.0..=1.0).contains(&f), "{s:?}");
    }
    let hybrids = (s.hybrid_frac * s.n as f64).round() as usize;
    assert!(s.hybrid_keys <= hybrids, "{s:?}");
    assert_eq!(s, prior_stats(&decode, 2, &judge, SAMPLE_SEED, 1000));
}

#[test]
fn prior_line_has_the_exact_format() {
    let (ds, sep, vae) = model();
    let judge = StyleJudge::new(ds, &vae.reconstruct_mean(&ds.x));
    let decode = |z: &ndarray::Array2<f64>| vae.decode(z);
    let s = prior_stats(&decode, 2, &judge, SAMPLE_SEED, 1000);
    let line = prior_line(&s, sep);
    assert!(line.starts_with("PRIOR: de 1000 puntos"), "{line}");
    assert!(line.contains("a una cara plausible (marco intacto y cada region en un estilo o entre dos)"), "{line}");
    assert!(line.contains("a un hibrido plausible"), "{line}");
    assert!(line.ends_with("combinaciones distintas"), "{line}");
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

    let dir2 = unique_tmp_dir("new-emoji-empty");
    let err = export_new_emoji(&empty_search(10), ds, &dir2).unwrap_err().to_string();
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
