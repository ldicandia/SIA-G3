//! Denoising Autoencoder (Ej 1b): trazador, determinismo y garantias de la evaluacion.

use std::collections::HashSet;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use tp5::autoencoder::{ArchSpec, Autoencoder};
use tp5::data::{load_font, N_PIXELS};
use tp5::denoising::{
    base_dae_config, evaluate_level, max_level_line, max_removed_level, train_dae, train_noise_seed, DaeConfig,
    LevelStats, N_EVAL_SAMPLES, TEST_NOISE_SEED,
};
use tp5::mlp::{Activation, Loss, Mlp};
use tp5::noise::{distort, level_rng, tile_rows, validate_levels, NoiseSchedule, TrainNoiseStream};
use tp5::optimizer::OptimizerKind;

fn assert_same_net(a: &Mlp, b: &Mlp) {
    assert_eq!(a.layers.len(), b.layers.len());
    for (la, lb) in a.layers.iter().zip(&b.layers) {
        assert!(la.w.iter().zip(lb.w.iter()).all(|(x, y)| x.to_bits() == y.to_bits()), "pesos distintos");
        assert!(la.b.iter().zip(lb.b.iter()).all(|(x, y)| x.to_bits() == y.to_bits()), "sesgos distintos");
    }
}

fn same_weights(a: &Autoencoder, b: &Autoencoder) -> bool {
    let eq = |x: &Mlp, y: &Mlp| {
        x.layers.iter().zip(&y.layers).all(|(la, lb)| {
            la.w.iter().zip(lb.w.iter()).all(|(p, q)| p.to_bits() == q.to_bits())
                && la.b.iter().zip(lb.b.iter()).all(|(p, q)| p.to_bits() == q.to_bits())
        })
    };
    eq(&a.encoder, &b.encoder) && eq(&a.decoder, &b.decoder)
}

fn short_cfg(epochs: usize) -> DaeConfig {
    let mut cfg = base_dae_config();
    cfg.epochs = epochs;
    cfg.copies = 2;
    cfg
}

#[test]
fn dae_removes_noise_at_level_0_1() {
    let ds = load_font().unwrap();
    let mut cfg = base_dae_config();
    cfg.epochs = 1000;
    cfg.copies = 4;
    let (ae, _) = train_dae(&ds, &cfg);
    let s = evaluate_level(&ae, &ds.x, 0.10, N_EVAL_SAMPLES, TEST_NOISE_SEED);
    assert!(
        s.output_mean_wrong < s.input_mean_wrong / 2.0,
        "salida {} no es menos de la mitad de la entrada {}",
        s.output_mean_wrong,
        s.input_mean_wrong
    );
    assert!(
        (s.input_mean_wrong - 3.5).abs() < 0.35,
        "la entrada deberia tener cerca de 35 * 0.10 pixeles incorrectos, tiene {}",
        s.input_mean_wrong
    );
}

#[test]
fn evaluation_is_deterministic() {
    let ds = load_font().unwrap();
    let (ae, _) = train_dae(&ds, &short_cfg(20));
    let a = evaluate_level(&ae, &ds.x, 0.20, 20, TEST_NOISE_SEED);
    let b = evaluate_level(&ae, &ds.x, 0.20, 20, TEST_NOISE_SEED);
    assert_eq!(a, b);
    assert_eq!(a.input_mean_wrong.to_bits(), b.input_mean_wrong.to_bits());
    let c = evaluate_level(&ae, &ds.x, 0.20, 20, TEST_NOISE_SEED ^ 1);
    assert_ne!(a.input_mean_wrong, c.input_mean_wrong);
}

#[test]
fn train_dae_is_deterministic_by_seed() {
    let ds = load_font().unwrap();
    let cfg = short_cfg(50);
    let (a, ha) = train_dae(&ds, &cfg);
    let (b, hb) = train_dae(&ds, &cfg);
    assert_same_net(&a.encoder, &b.encoder);
    assert_same_net(&a.decoder, &b.decoder);
    assert_eq!(ha.loss.len(), 50);
    assert!(ha.loss.iter().zip(&hb.loss).all(|(x, y)| x.to_bits() == y.to_bits()));

    let mut other = cfg.clone();
    other.seed = 1;
    let (c, _) = train_dae(&ds, &other);
    assert!(!same_weights(&a, &c), "otra semilla deberia dar otros pesos");
}

fn stats_with(level: f64, frac_within: f64) -> LevelStats {
    LevelStats {
        level,
        n_samples: 100,
        input_mean_wrong: 0.0,
        output_mean_wrong: 0.0,
        output_max_wrong: 0,
        input_frac_within: 1.0,
        frac_within,
    }
}

#[test]
fn level_zero_training_equals_plain_autoencoder() {
    let ds = load_font().unwrap();
    let spec = ArchSpec {
        hidden: vec![16],
        latent_dim: 2,
        hidden_act: Activation::Tanh,
        latent_act: Activation::Identity,
        output_act: Activation::Sigmoid,
    };
    let cfg = DaeConfig {
        name: "nivel0-s7".to_string(),
        study: "check".to_string(),
        spec: spec.clone(),
        loss: Loss::Bce,
        optimizer: OptimizerKind::Adam { lr: 0.005, beta1: 0.9, beta2: 0.999, eps: 1e-8 },
        train_noise: NoiseSchedule::Fixed(0.0),
        epochs: 30,
        copies: 1,
        seed: 7,
    };
    let (dae, _) = train_dae(&ds, &cfg);

    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let mut plain = Autoencoder::new(&spec, N_PIXELS, &mut rng);
    let mut opt_enc = cfg.optimizer.build(&plain.encoder);
    let mut opt_dec = cfg.optimizer.build(&plain.decoder);
    for _ in 0..30 {
        plain.train_step(&ds.x, &ds.x, cfg.loss, opt_enc.as_mut(), opt_dec.as_mut());
    }
    assert_same_net(&dae.encoder, &plain.encoder);
    assert_same_net(&dae.decoder, &plain.decoder);
}

#[test]
fn evaluation_at_level_zero_has_no_input_error() {
    let ds = load_font().unwrap();
    let (ae, _) = train_dae(&ds, &short_cfg(10));
    let s = evaluate_level(&ae, &ds.x, 0.0, 10, TEST_NOISE_SEED);
    assert_eq!(s.input_mean_wrong, 0.0);
    assert_eq!(s.input_frac_within, 1.0);
}

#[test]
fn evaluation_noise_is_unseen_in_training() {
    let ds = load_font().unwrap();
    let mut stream = TrainNoiseStream::new(NoiseSchedule::Uniform { lo: 0.0, hi: 0.4 }, train_noise_seed(0));
    let mut seen: HashSet<(usize, Vec<u8>)> = HashSet::new();
    for _ in 0..300 {
        let batch = stream.next_batch(&ds.x);
        for (i, row) in batch.rows().into_iter().enumerate() {
            seen.insert((i, row.iter().map(|&v| v as u8).collect()));
        }
    }
    let clean = tile_rows(&ds.x, N_EVAL_SAMPLES);
    let noisy = distort(&clean, 0.20, &mut level_rng(TEST_NOISE_SEED, 0.20));
    let n = noisy.nrows();
    let repeated = noisy
        .rows()
        .into_iter()
        .enumerate()
        .filter(|(i, row)| seen.contains(&(i % ds.x.nrows(), row.iter().map(|&v| v as u8).collect())))
        .count();
    assert!(
        (repeated as f64) < 0.01 * n as f64,
        "{repeated} de {n} imagenes de evaluacion aparecen tal cual en el entrenamiento"
    );
    for s in 0..64u64 {
        assert_ne!(train_noise_seed(s), TEST_NOISE_SEED);
    }
}

#[test]
fn max_removed_level_prefix_semantics() {
    let st = |fr: &[f64]| -> Vec<LevelStats> {
        fr.iter().enumerate().map(|(i, &f)| stats_with(0.1 * i as f64, f)).collect()
    };
    let s = st(&[1.0, 0.95, 0.5, 0.95]);
    assert_eq!(max_removed_level(&s), Some(0.1));
    assert_eq!(max_removed_level(&st(&[0.5, 1.0, 1.0])), None);
    assert_eq!(max_removed_level(&[]), None);
    let mut shuffled = s.clone();
    shuffled.reverse();
    shuffled.swap(0, 2);
    assert_eq!(max_removed_level(&shuffled), max_removed_level(&s));
}

#[test]
fn max_level_line_formats_both_outcomes() {
    let some = max_level_line(&[stats_with(0.0, 1.0), stats_with(0.1, 0.95)]);
    let none = max_level_line(&[stats_with(0.0, 0.2)]);
    assert!(some.starts_with("NIVEL MAXIMO ELIMINADO:"), "{some}");
    assert!(none.starts_with("NIVEL MAXIMO ELIMINADO:"), "{none}");
    assert!(some.contains("0.10"), "{some}");
    assert!(none.contains("ninguno"), "{none}");
}

#[test]
fn validate_levels_rejects_bad_lists() {
    assert!(validate_levels(&[]).is_err());
    assert!(validate_levels(&[f64::NAN]).is_err());
    assert!(validate_levels(&[-0.1]).is_err());
    assert!(validate_levels(&[0.5000001]).is_err());
    assert!(validate_levels(&[0.1, 0.1]).is_err());
    assert!(validate_levels(&[0.1, 0.2, 0.1]).is_err());
    let thirteen: Vec<f64> = (0..13).map(|i| i as f64 * 0.03).collect();
    assert!(validate_levels(&thirteen).is_err());
    assert_eq!(validate_levels(&[0.4, 0.0, 0.2]).unwrap(), vec![0.0, 0.2, 0.4]);
    assert!(validate_levels(&[0.0, 0.5]).is_ok());
    let twelve: Vec<f64> = (0..12).map(|i| i as f64 * 0.04).collect();
    assert!(validate_levels(&twelve).is_ok());
}
