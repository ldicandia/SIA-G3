//! Verificacion de backprop contra diferencias finitas centrales a traves de la cadena
//! encoder -> latente -> decoder.

use ndarray::Array2;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use tp5::autoencoder::{ArchSpec, Autoencoder};
use tp5::mlp::{Activation, Loss};

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

fn layer_mut(ae: &mut Autoencoder, is_enc: bool, l: usize) -> &mut tp5::mlp::DenseLayer {
    if is_enc {
        &mut ae.encoder.layers[l]
    } else {
        &mut ae.decoder.layers[l]
    }
}

fn nudge(ae: &mut Autoencoder, is_enc: bool, l: usize, p: Param, delta: f64) {
    let layer = layer_mut(ae, is_enc, l);
    match p {
        Param::W(i, j) => layer.w[[i, j]] += delta,
        Param::B(j) => layer.b[j] += delta,
    }
}

/// Diferencia central del valor de la perdida respecto de un parametro.
fn numeric(
    ae: &mut Autoencoder,
    is_enc: bool,
    l: usize,
    p: Param,
    input: &Array2<f64>,
    target: &Array2<f64>,
    loss: Loss,
) -> f64 {
    nudge(ae, is_enc, l, p, H);
    let plus = ae.loss_and_grads(input, target, loss).0;
    nudge(ae, is_enc, l, p, -2.0 * H);
    let minus = ae.loss_and_grads(input, target, loss).0;
    nudge(ae, is_enc, l, p, H);
    (plus - minus) / (2.0 * H)
}

/// Compara el gradiente analitico de cada parametro contra la diferencia central.
fn check(loss: Loss, hidden_act: Activation, latent_act: Activation, seed: u64) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let spec = ArchSpec {
        hidden: vec![5],
        latent_dim: 2,
        hidden_act,
        latent_act,
        output_act: Activation::Sigmoid,
    };
    let mut ae = Autoencoder::new(&spec, 6, &mut rng);
    let target = Array2::from_shape_fn((4, 6), |_| if rng.gen_bool(0.5) { 1.0 } else { 0.0 });
    let input = target.clone();

    let (_, grads) = ae.loss_and_grads(&input, &target, loss);
    let mut checked = 0;
    let mut worst = 0.0_f64;

    for is_enc in [true, false] {
        let g = if is_enc { &grads.enc } else { &grads.dec };
        let n_layers = g.dw.len();
        for l in 0..n_layers {
            let (rows, cols) = g.dw[l].dim();
            for i in 0..rows {
                for j in 0..cols {
                    let n = numeric(&mut ae, is_enc, l, Param::W(i, j), &input, &target, loss);
                    let a = g.dw[l][[i, j]];
                    let e = rel_err(a, n);
                    worst = worst.max(e);
                    assert!(e < TOL, "w[{l}][{i},{j}] enc={is_enc}: analitico {a} numerico {n}");
                    checked += 1;
                }
            }
            for j in 0..cols {
                let n = numeric(&mut ae, is_enc, l, Param::B(j), &input, &target, loss);
                let a = g.db[l][j];
                let e = rel_err(a, n);
                worst = worst.max(e);
                assert!(e < TOL, "b[{l}][{j}] enc={is_enc}: analitico {a} numerico {n}");
                checked += 1;
            }
        }
    }
    // 6*5+5 + 5*2+2 + 2*5+5 + 5*6+6 parametros
    assert_eq!(checked, 98);
    println!("peor error relativo: {worst:e}");
}

#[test]
fn mse_sigmoid_tanh_identity() {
    check(Loss::Mse, Activation::Tanh, Activation::Identity, 1);
}

#[test]
fn bce_sigmoid_tanh_identity() {
    check(Loss::Bce, Activation::Tanh, Activation::Identity, 2);
}

#[test]
fn bce_sigmoid_relu_tanh() {
    check(Loss::Bce, Activation::Relu, Activation::Tanh, 3);
}

#[test]
fn mse_sigmoid_sigmoid_identity() {
    check(Loss::Mse, Activation::Sigmoid, Activation::Identity, 4);
}
