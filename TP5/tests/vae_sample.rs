//! Busqueda semillada de bloques nuevos en el prior (deduplicada por combinacion y ordenada por
//! equilibrio de la mezcla), galeria, estadisticas del prior y exportacion de la figura.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::OnceLock;

use tp5::ores::{load_ores, separation, OreDataset, ORE_MINERALS, ORE_ROCKS};
use tp5::vae::{chosen_vae_config, train_vae, Vae, SAMPLE_SEED};
use tp5::vae_eval::{
    new_ore_line, prior_line, prior_stats, render_gallery, search_new_ore, validate_draws, JudgeFloors, OreJudge,
    SampleSearch, DEFAULT_DRAWS, GALLERY_SIZE, MIN_BACKGROUND, MIN_SHARE, RESIDUAL_FRACTION,
};
use tp5::vae_figures::{build_new_ore_tiles, export_new_ore, NEW_ORE_PNG};

const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

fn unique_tmp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tp5-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// El VAE de la configuracion congelada, entrenado una sola vez para todo el archivo.
fn model() -> &'static (OreDataset, Vae, OreJudge) {
    static MODEL: OnceLock<(OreDataset, Vae, OreJudge)> = OnceLock::new();
    MODEL.get_or_init(|| {
        let ds = ores();
        let (vae, _) = train_vae(&ds, &chosen_vae_config()).unwrap();
        let judge = OreJudge::new(&ds, &vae.reconstruct_mean(&ds.x));
        (ds, vae, judge)
    })
}

fn default_search() -> &'static SampleSearch {
    static S: OnceLock<SampleSearch> = OnceLock::new();
    S.get_or_init(|| {
        let (ds, vae, _) = model();
        search_new_ore(vae, ds, SAMPLE_SEED, DEFAULT_DRAWS)
    })
}

fn empty_search(max_draws: usize) -> SampleSearch {
    SampleSearch {
        seed: 1,
        max_draws,
        examined: max_draws,
        floors: JudgeFloors { background: 0, residual: 0.0, foreign_share: 0.0, contrast: 0.0 },
        residual_limit: 0.0,
        candidates: 0,
        distinct_keys: 0,
        accepted: vec![],
    }
}

#[test]
fn search_is_deterministic_by_seed() {
    let (ds, vae, _) = model();
    let a = default_search();
    let b = search_new_ore(vae, ds, SAMPLE_SEED, DEFAULT_DRAWS);
    assert_eq!(a.examined, b.examined);
    assert_eq!(a.candidates, b.candidates);
    assert_eq!(a.distinct_keys, b.distinct_keys);
    assert_eq!(a.floors, b.floors);
    assert_eq!(a.accepted.len(), b.accepted.len());
    for (p, q) in a.accepted.iter().zip(&b.accepted) {
        assert_eq!(p.draw_index, q.draw_index);
        assert!(p.z.iter().zip(q.z.iter()).all(|(u, v)| u.to_bits() == v.to_bits()));
        assert!(p.image.iter().zip(q.image.iter()).all(|(u, v)| u.to_bits() == v.to_bits()));
        assert_eq!(p.verdict, q.verdict);
    }
    let c = search_new_ore(vae, ds, SAMPLE_SEED ^ 1, DEFAULT_DRAWS);
    assert!(!a.accepted.is_empty() && !c.accepted.is_empty());
    assert_ne!(a.accepted[0].z, c.accepted[0].z);
}

#[test]
fn longer_search_never_finds_fewer_combinations() {
    let (ds, vae, _) = model();
    let short = search_new_ore(vae, ds, SAMPLE_SEED, 300);
    let long = default_search();
    assert!(long.examined > short.examined);
    assert!(long.candidates >= short.candidates);
    assert!(long.distinct_keys >= short.distinct_keys);
}

#[test]
fn gallery_is_deduplicated_and_ranked_by_balance() {
    let search = default_search();
    assert!(!search.accepted.is_empty());
    assert!(search.accepted.len() <= GALLERY_SIZE);
    let keys: HashSet<String> = search.accepted.iter().map(|s| s.verdict.combination()).collect();
    assert_eq!(keys.len(), search.accepted.len(), "hay combinaciones repetidas en la galeria");
    for w in search.accepted.windows(2) {
        let (p, q) = (w[0].verdict.balance(), w[1].verdict.balance());
        assert!(p >= q, "equilibrio no decreciente: {p} luego {q}");
        if p == q {
            assert!(w[0].verdict.residual <= w[1].verdict.residual, "empates ordenados por residuo");
        }
    }
    assert!(search.distinct_keys >= search.accepted.len());
    assert!(search.candidates >= search.distinct_keys);
}

#[test]
fn accepted_samples_are_new_blocks_verified_independently() {
    let (ds, _, judge) = model();
    let search = default_search();
    let sep = separation(ds);
    let known: HashSet<(usize, usize)> = ds.combinations();
    for s in &search.accepted {
        // roca: textura base con mas pixeles a menos de 32 niveles en los tres canales, recontada con bucles
        let mut counts = [0usize; 3];
        for (r, count) in counts.iter_mut().enumerate() {
            for p in 0..256 {
                if (0..3).all(|c| (s.image[3 * p + c] - ds.bases[[r, 3 * p + c]]).abs() <= 32.0 / 255.0) {
                    *count += 1;
                }
            }
        }
        let rock = (0..3).fold(0, |best, r| if counts[r] > counts[best] { r } else { best });
        assert_eq!(s.verdict.rock, rock);
        assert_eq!(s.verdict.background, counts[rock]);
        assert!(counts[rock] >= MIN_BACKGROUND);
        // minerales: pesos de la mezcla sumados por mineral
        assert!((s.verdict.shares.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        let present: Vec<usize> = (0..ORE_MINERALS.len()).filter(|&m| s.verdict.shares[m] >= MIN_SHARE).collect();
        assert_eq!(s.verdict.minerals, present);
        assert!(s.verdict.residual <= RESIDUAL_FRACTION * sep.min);
        assert!(s.verdict.contrast >= search.floors.contrast && search.floors.contrast > 0.0);
        // la combinacion no existe en el dataset: dos o mas minerales, o un mineral en una roca ajena
        let single_known = present.len() == 1 && known.contains(&(rock, present[0]));
        assert!(!single_known, "{} es una combinacion del dataset", s.verdict.combination());
        assert!(s.verdict.plausible && s.verdict.new_block && !s.verdict.known);
        assert_eq!(judge.judge(s.image.view()), s.verdict);
    }
}

#[test]
fn search_finds_a_full_gallery_within_the_default_draws() {
    let search = default_search();
    assert_eq!(search.accepted.len(), GALLERY_SIZE, "solo {} combinaciones distintas", search.distinct_keys);
}

#[test]
fn new_ore_line_formats_success_and_failure() {
    let (ds, _, _) = model();
    let search = default_search();
    let line = new_ore_line(search, ds);
    let h = &search.accepted[0];
    assert!(line.starts_with("BLOQUE NUEVO: z=("), "{line}");
    assert!(line.contains(&format!("| {} (combinacion ausente del dataset) |", h.verdict.combination())), "{line}");
    assert!(line.contains(&format!("de {} (minimo ", ORE_ROCKS[h.verdict.rock])), "{line}");
    assert!(line.contains(&format!("bloque mas cercano: {} a ", ds.names[h.nearest.0])), "{line}");
    assert!(line.contains(" | criterio: "), "{line}");
    let none = new_ore_line(&empty_search(50), ds);
    assert!(none.starts_with("BLOQUE NUEVO: ninguno en 50 sorteos"), "{none}");
}

#[test]
fn gallery_lines_have_the_exact_format() {
    let (ds, _, _) = model();
    let search = default_search();
    let text = render_gallery(search, ds);
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 1 + search.accepted.len());
    assert!(lines[0].starts_with("GALERIA: "), "{}", lines[0]);
    for (i, l) in lines.iter().enumerate().skip(1) {
        let s = &search.accepted[i - 1];
        assert!(l.starts_with(&format!("  nuevo {i}: sorteo {} | {} | mezcla: ", s.draw_index + 1, s.verdict.combination())), "{l}");
        assert!(l.contains(" | residuo "), "{l}");
    }
}

#[test]
fn prior_stats_are_consistent_and_deterministic() {
    let (_, vae, judge) = model();
    let decode = |z: &ndarray::Array2<f64>| vae.decode(z);
    let s = prior_stats(&decode, 2, judge, SAMPLE_SEED, 1000);
    assert_eq!(s.n, 1000);
    assert!((s.recognizable_frac + s.new_frac - s.plausible_frac).abs() < 1e-12, "{s:?}");
    for f in [s.recognizable_frac, s.plausible_frac, s.new_frac] {
        assert!((0.0..=1.0).contains(&f), "{s:?}");
    }
    let news = (s.new_frac * s.n as f64).round() as usize;
    assert!(s.new_keys <= news, "{s:?}");
    assert_eq!(s, prior_stats(&decode, 2, judge, SAMPLE_SEED, 1000));
    // la busqueda recorre el mismo flujo: acepta exactamente los nuevos del prior
    assert_eq!(default_search().candidates, news);
    assert_eq!(default_search().distinct_keys, s.new_keys);
}

#[test]
fn prior_line_has_the_exact_format() {
    let (_, vae, judge) = model();
    let decode = |z: &ndarray::Array2<f64>| vae.decode(z);
    let s = prior_stats(&decode, 2, judge, SAMPLE_SEED, 1000);
    let line = prior_line(&s);
    assert!(line.starts_with("PRIOR: de 1000 puntos"), "{line}");
    assert!(line.contains("a un bloque plausible (fondo de una roca, mezcla de bloques del dataset y contraste de un bloque real)"), "{line}");
    assert!(line.contains("a un bloque nuevo (plausible y con una combinacion ausente)"), "{line}");
    assert!(line.ends_with("combinaciones distintas"), "{line}");
}

#[test]
fn new_ore_png_is_written_with_the_fixed_name_and_fails_when_empty() {
    let search = default_search();
    let tiles = build_new_ore_tiles(search);
    assert_eq!(tiles.len(), search.accepted.len());
    let first: Vec<&str> = search.accepted[0].verdict.minerals.iter().map(|&m| ORE_MINERALS[m]).collect();
    assert_eq!(tiles[0].0, format!("1: {}", first.join("+")));

    let dir = unique_tmp_dir("new-bloque");
    let path = export_new_ore(search, &dir).unwrap();
    assert_eq!(path, dir.join(NEW_ORE_PNG));
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..8], &PNG_SIGNATURE);
    assert!(bytes.len() > 3000);

    let dir2 = unique_tmp_dir("new-bloque-empty");
    let err = export_new_ore(&empty_search(10), &dir2).unwrap_err().to_string();
    assert!(err.contains("10 sorteos"), "{err}");
    assert!(!dir2.join(NEW_ORE_PNG).exists());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn validate_draws_bounds() {
    assert!(validate_draws(0).is_err());
    assert!(validate_draws(100_001).is_err());
    assert_eq!(validate_draws(1).unwrap(), 1);
    assert_eq!(validate_draws(100_000).unwrap(), 100_000);
}

/// El dataset de menas; falta si no se corrio `bash scripts/descargar_menas.sh`.
fn ores() -> OreDataset {
    load_ores().expect("dataset de menas: corra `bash scripts/descargar_menas.sh`")
}
