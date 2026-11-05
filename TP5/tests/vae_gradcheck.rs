//! Verificacion de los gradientes analiticos del VAE contra diferencias finitas centrales, con el
//! ruido `eps` fijo, a traves de encoder -> media/log-varianza -> reparametrizacion -> decoder.

use ndarray::Array2;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use tp5::autoencoder::ArchSpec;
use tp5::mlp::{Activation, DenseLayer, Loss};
use tp5::vae::{kl_divergence, reparameterize, standard_normal_matrix, Vae};

const H: f64 = 1e-5;
const TOL: f64 = 1e-5;

#[derive(Clone, Copy)]
enum Param {
    W(usize, usize),
    B(usize),
}

fn rel_err(a: f64, n: f64) -> f64 {
    (a - n).abs() / (a.abs() + n.abs()).max(1e-8)
}

fn layer_mut(vae: &mut Vae, is_enc: bool, l: usize) -> &mut DenseLayer {
    if is_enc {
        &mut vae.encoder.layers[l]
    } else {
        &mut vae.decoder.layers[l]
    }
}

fn nudge(vae: &mut Vae, is_enc: bool, l: usize, p: Param, delta: f64) {
    let layer = layer_mut(vae, is_enc, l);
    match p {
        Param::W(i, j) => layer.w[[i, j]] += delta,
        Param::B(j) => layer.b[j] += delta,
    }
}

fn total(vae: &Vae, x: &Array2<f64>, eps: &Array2<f64>, loss: Loss) -> f64 {
    vae.loss_and_grads(x, eps, loss).0.total()
}

fn numeric(vae: &mut Vae, is_enc: bool, l: usize, p: Param, x: &Array2<f64>, eps: &Array2<f64>, loss: Loss) -> f64 {
    nudge(vae, is_enc, l, p, H);
    let plus = total(vae, x, eps, loss);
    nudge(vae, is_enc, l, p, -2.0 * H);
    let minus = total(vae, x, eps, loss);
    nudge(vae, is_enc, l, p, H);
    (plus - minus) / (2.0 * H)
}

fn tiny_vae(seed: u64) -> (Vae, Array2<f64>, Array2<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let spec = ArchSpec {
        hidden: vec![5],
        latent_dim: 2,
        hidden_act: Activation::Tanh,
        latent_act: Activation::Identity,
        output_act: Activation::Sigmoid,
    };
    let vae = Vae::new(&spec, 6, &mut rng);
    let x = Array2::from_shape_fn((4, 6), |_| if rng.gen_bool(0.5) { 1.0 } else { 0.0 });
    let eps = standard_normal_matrix(&mut rng, 4, 2);
    (vae, x, eps)
}

fn check(loss: Loss, seed: u64) {
    let (mut vae, x, eps) = tiny_vae(seed);
    let (_, grads) = vae.loss_and_grads(&x, &eps, loss);
    let mut checked = 0;
    let mut worst = 0.0_f64;

    for is_enc in [true, false] {
        let g = if is_enc { &grads.enc } else { &grads.dec };
        for l in 0..g.dw.len() {
            let (rows, cols) = g.dw[l].dim();
            for i in 0..rows {
                for j in 0..cols {
                    let n = numeric(&mut vae, is_enc, l, Param::W(i, j), &x, &eps, loss);
                    let a = g.dw[l][[i, j]];
                    let e = rel_err(a, n);
                    worst = worst.max(e);
                    assert!(e < TOL, "w[{l}][{i},{j}] enc={is_enc}: analitico {a} numerico {n}");
                    checked += 1;
                }
            }
            for j in 0..cols {
                let n = numeric(&mut vae, is_enc, l, Param::B(j), &x, &eps, loss);
                let a = g.db[l][j];
                let e = rel_err(a, n);
                worst = worst.max(e);
                assert!(e < TOL, "b[{l}][{j}] enc={is_enc}: analitico {a} numerico {n}");
                checked += 1;
            }
        }
    }
    // codificador 6*5+5 + 5*4+4, decodificador 2*5+5 + 5*6+6
    assert_eq!(checked, 6 * 5 + 5 + 5 * 4 + 4 + 2 * 5 + 5 + 5 * 6 + 6);
    assert_eq!(checked, 110);
    println!("peor error relativo: {worst:e}");
}

#[test]
fn bce_tanh_gradients_match_finite_differences() {
    check(Loss::Bce, 11);
}

#[test]
fn mse_tanh_gradients_match_finite_differences() {
    check(Loss::Mse, 12);
}

#[test]
fn total_equals_recon_plus_kl() {
    for loss in [Loss::Bce, Loss::Mse] {
        let (vae, x, eps) = tiny_vae(13);
        let (terms, _) = vae.loss_and_grads(&x, &eps, loss);
        let (mu, logvar) = vae.encode_params(&x);
        let z = reparameterize(&mu, &logvar, &eps);
        let pred = vae.decode(&z);
        let recon = loss.value(&pred, &x);
        let kl = kl_divergence(&mu, &logvar);
        assert!((terms.recon - recon).abs() < 1e-12);
        assert!((terms.kl - kl).abs() < 1e-12);
        assert!((terms.total() - (recon + kl)).abs() < 1e-12);
    }
}
