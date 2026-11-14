//! Aceptacion del resultado generativo del VAE con la configuracion congelada en tres semillas y
//! contraste con el autoencoder simple.

use std::sync::OnceLock;

use tp5::autoencoder::Autoencoder;
use tp5::ores::{load_ores, OreDataset};
use tp5::vae::{chosen_vae_config, reconstruction_summary, train_vae, Vae, VaeHistory, SAMPLE_SEED};
use tp5::vae_eval::{
    acceptance_checks, contrast_line, control_line, decode_grid, manifold_metrics, mean_wrong_pixels,
    negative_control, plain_grid_points, prior_stats, render_acceptance, search_new_ore, train_plain_ae,
    AcceptanceCheck, Contrast, OreJudge, PriorStats, CONTROL_POSTERIOR_SAMPLES, DEFAULT_DRAWS, GRID_HALF_RANGE,
    GRID_SIDE_DEFAULT, MAX_MEAN_WRONG_PX, MIN_VALID_GAP,
};

struct Trained {
    ds: OreDataset,
    models: Vec<(Vae, VaeHistory)>,
    plain: Autoencoder,
}

fn trained() -> &'static Trained {
    static T: OnceLock<Trained> = OnceLock::new();
    T.get_or_init(|| {
        let ds = ores();
        let (models, plain) = std::thread::scope(|s| {
            let handles: Vec<_> = (0..3u64)
                .map(|seed| {
                    let ds = &ds;
                    s.spawn(move || {
                        let mut cfg = chosen_vae_config();
                        cfg.seed = seed;
                        train_vae(ds, &cfg).unwrap()
                    })
                })
                .collect();
            let plain_handle = s.spawn(|| train_plain_ae(&ds, &chosen_vae_config()));
            let models: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
            (models, plain_handle.join().unwrap())
        });
        Trained { ds, models, plain }
    })
}

fn plain_judge(t: &Trained) -> OreJudge {
    OreJudge::new(&t.ds, &t.plain.reconstruct(&t.ds.x))
}

fn vae_judge(t: &Trained, seed: usize) -> OreJudge {
    OreJudge::new(&t.ds, &t.models[seed].0.reconstruct_mean(&t.ds.x))
}

fn plain_prior(t: &Trained) -> PriorStats {
    let decode = |z: &ndarray::Array2<f64>| t.plain.decode(z);
    prior_stats(&decode, 2, &plain_judge(t), SAMPLE_SEED, DEFAULT_DRAWS)
}

fn vae_prior(t: &Trained, seed: usize) -> PriorStats {
    let vae = &t.models[seed].0;
    let decode = |z: &ndarray::Array2<f64>| vae.decode(z);
    prior_stats(&decode, 2, &vae_judge(t, seed), SAMPLE_SEED, DEFAULT_DRAWS)
}

#[test]
fn acceptance_holds_for_seeds_0_1_2() {
    let t = trained();
    let plain = plain_prior(t);
    let mut failures = Vec::new();
    for seed in 0..3 {
        let (vae, hist) = &t.models[seed];
        let judge = vae_judge(t, seed);
        let recon = reconstruction_summary(vae, &t.ds);
        let grid =
            manifold_metrics(&decode_grid(vae, GRID_SIDE_DEFAULT, GRID_HALF_RANGE), GRID_SIDE_DEFAULT, &judge);
        let prior = vae_prior(t, seed);
        let control = negative_control(vae, &judge);
        let checks = acceptance_checks(&recon, hist, &grid, &prior, &plain, &control, &judge);
        assert_eq!(checks.len(), 11);
        println!("semilla {seed}:\n{}", render_acceptance(&checks));
        for c in checks.iter().filter(|c| !c.pass) {
            failures.push(format!("semilla {seed}: {} = {:.3}", c.name, c.value));
        }
    }
    assert!(failures.is_empty(), "verificaciones que fallan: {failures:?}");
}

#[test]
fn negative_control_recognizes_every_training_reconstruction_with_its_own_combination() {
    let t = trained();
    for seed in 0..3 {
        let c = negative_control(&t.models[seed].0, &vae_judge(t, seed));
        println!("semilla {seed}: {}", control_line(&c));
        assert_eq!(c.n_mean, 18);
        assert_eq!(c.mean_accepted, 0, "semilla {seed}: {c:?}");
        assert_eq!(c.mean_own_combination, 18, "semilla {seed}: {c:?}");
    }
}

#[test]
fn posterior_samples_are_accepted_less_often_than_prior_draws() {
    let t = trained();
    for seed in 0..3 {
        let c = negative_control(&t.models[seed].0, &vae_judge(t, seed));
        assert_eq!(c.n_posterior, 18 * CONTROL_POSTERIOR_SAMPLES);
        let posterior = c.posterior_accepted as f64 / c.n_posterior as f64;
        let prior = vae_prior(t, seed).new_frac;
        println!("semilla {seed}: posterior {posterior:.3} contra prior {prior:.3}");
        assert!(posterior < prior, "semilla {seed}: posterior {posterior} no es menor que prior {prior}");
    }
}

#[test]
fn predicate_separates_vae_from_plain_through_plausibility() {
    let t = trained();
    let plain = plain_prior(t);
    for seed in 0..3 {
        let vae = vae_prior(t, seed);
        println!(
            "semilla {seed}: plausibles VAE {:.3} contra {:.3} | nuevos VAE {:.3} contra {:.3}",
            vae.plausible_frac, plain.plausible_frac, vae.new_frac, plain.new_frac
        );
        assert!(
            vae.plausible_frac - plain.plausible_frac >= MIN_VALID_GAP,
            "semilla {seed}: ventaja de plausibilidad {}",
            vae.plausible_frac - plain.plausible_frac
        );
    }
}

#[test]
fn control_line_format() {
    let t = trained();
    let c = negative_control(&t.models[0].0, &vae_judge(t, 0));
    let line = control_line(&c);
    assert!(
        line.starts_with("CONTROL NEGATIVO: reconstrucciones de los 18 bloques del entrenamiento aceptadas como nuevas: "),
        "{line}"
    );
    assert!(line.contains("(reconocidas con su propia combinacion: 18/18)"), "{line}");
    assert!(line.ends_with(" % aceptadas como nuevas"), "{line}");
}

#[test]
fn chosen_seed_finds_a_new_ore_within_the_default_draws() {
    let t = trained();
    let search = search_new_ore(&t.models[0].0, &t.ds, SAMPLE_SEED, DEFAULT_DRAWS);
    assert!(!search.accepted.is_empty(), "ningun bloque nuevo en {DEFAULT_DRAWS} sorteos");
}

#[test]
fn plain_autoencoder_reconstructs_but_samples_the_prior_worse() {
    let t = trained();
    let plain_recon = mean_wrong_pixels(&t.ds, &t.plain.reconstruct(&t.ds.x));
    assert!(plain_recon <= MAX_MEAN_WRONG_PX, "reconstruccion del autoencoder simple {plain_recon}");
    let vae_recon = reconstruction_summary(&t.models[0].0, &t.ds).mean_wrong;
    assert!(vae_recon <= MAX_MEAN_WRONG_PX);
    let (vae, plain) = (vae_prior(t, 0), plain_prior(t));
    println!(
        "sorteos plausibles: VAE {:.3}, autoencoder simple {:.3} | reconocibles: VAE {:.3}, {:.3} | nuevos: VAE {:.3}, {:.3}",
        vae.plausible_frac,
        plain.plausible_frac,
        vae.recognizable_frac,
        plain.recognizable_frac,
        vae.new_frac,
        plain.new_frac
    );
    assert!(vae.recognizable_frac > plain.recognizable_frac);
    assert!(vae.new_keys > plain.new_keys);
}

#[test]
fn plain_grid_covers_the_padded_code_box() {
    let t = trained();
    let codes = t.plain.encode(&t.ds.x);
    let col = |a: usize| -> (f64, f64) {
        let v: Vec<f64> = codes.column(a).to_vec();
        (v.iter().cloned().fold(f64::INFINITY, f64::min), v.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
    };
    let ((x0, x1), (y0, y1)) = (col(0), col(1));
    let (px, py) = (0.1 * (x1 - x0), 0.1 * (y1 - y0));
    let side = 7;
    let g = plain_grid_points(&t.plain, &t.ds, side);
    assert_eq!(g.dim(), (side * side, 2));
    let first = g.row(0);
    let last = g.row(side * side - 1);
    assert!((first[0] - (x0 - px)).abs() < 1e-12 && (first[1] - (y1 + py)).abs() < 1e-12, "{first:?}");
    assert!((last[0] - (x1 + px)).abs() < 1e-12 && (last[1] - (y0 - py)).abs() < 1e-12, "{last:?}");
    // z1 crece hacia la derecha y z2 decrece hacia abajo.
    assert!(g[[1, 0]] > g[[0, 0]]);
    assert!(g[[side, 1]] < g[[0, 1]]);
}

#[test]
fn contrast_line_format() {
    let t = trained();
    let vae = &t.models[0].0;
    let (vj, pj) = (vae_judge(t, 0), plain_judge(t));
    let c = Contrast {
        vae_prior: vae_prior(t, 0),
        plain_prior: plain_prior(t),
        vae_grid: manifold_metrics(&decode_grid(vae, 5, GRID_HALF_RANGE), 5, &vj),
        plain_grid: manifold_metrics(&t.plain.decode(&plain_grid_points(&t.plain, &t.ds, 5)), 5, &pj),
        vae_recon_mean: 5.0,
        plain_recon_mean: 6.0,
    };
    let line = contrast_line(&c);
    assert!(line.starts_with("CONTRASTE: puntos z ~ N(0, I) que decodifican a un bloque plausible: VAE "), "{line}");
    assert!(line.contains(" | nuevo: VAE "), "{line}");
    assert!(line.contains("reconstruccion media: VAE 5.00 px contra 6.00 px por bloque"), "{line}");
}

#[test]
fn render_acceptance_marks_pass_and_fail() {
    let checks = vec![
        AcceptanceCheck { name: "uno", value: 0.9, threshold: 0.5, at_least: true, pass: true },
        AcceptanceCheck { name: "dos", value: 20.0, threshold: 12.0, at_least: false, pass: false },
    ];
    let text = render_acceptance(&checks);
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 2);
    assert!(lines[0].starts_with("ACEPTACION OK:") && lines[0].contains(">="), "{}", lines[0]);
    assert!(lines[1].starts_with("ACEPTACION FALLA:") && lines[1].contains("<="), "{}", lines[1]);
}

/// El dataset de menas; falta si no se corrio `bash scripts/descargar_menas.sh`.
fn ores() -> OreDataset {
    load_ores().expect("dataset de menas: corra `bash scripts/descargar_menas.sh`")
}
