//! Perceptron multicapa con backpropagation manual sobre ndarray.

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_chacha::ChaCha8Rng;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    Identity,
    Tanh,
    Sigmoid,
    Relu,
}

/// Sigmoide estable para magnitudes grandes.
fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

impl Activation {
    pub fn apply(self, z: &Array2<f64>) -> Array2<f64> {
        match self {
            Activation::Identity => z.clone(),
            Activation::Tanh => z.mapv(f64::tanh),
            Activation::Sigmoid => z.mapv(sigmoid),
            Activation::Relu => z.mapv(|v| v.max(0.0)),
        }
    }

    /// Derivada expresada en funcion de la salida `a = f(z)`.
    pub fn derivative_from_output(self, a: &Array2<f64>) -> Array2<f64> {
        match self {
            Activation::Identity => Array2::ones(a.raw_dim()),
            Activation::Tanh => a.mapv(|v| 1.0 - v * v),
            Activation::Sigmoid => a.mapv(|v| v * (1.0 - v)),
            Activation::Relu => a.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DenseLayer {
    /// Pesos, forma `n_in x n_out`.
    pub w: Array2<f64>,
    /// Sesgos, largo `n_out`.
    pub b: Array1<f64>,
    pub act: Activation,
}

#[derive(Debug, Clone)]
pub struct Mlp {
    pub layers: Vec<DenseLayer>,
}

/// Activaciones de un forward: `a[0]` = entrada, `a[l + 1]` = salida de la capa `l`.
#[derive(Debug, Clone)]
pub struct Cache {
    pub a: Vec<Array2<f64>>,
}

impl Cache {
    pub fn last(&self) -> &Array2<f64> {
        self.a.last().expect("cache vacia")
    }
}

/// Gradientes por capa, en el mismo orden que `Mlp::layers`.
#[derive(Debug, Clone)]
pub struct Grads {
    pub dw: Vec<Array2<f64>>,
    pub db: Vec<Array1<f64>>,
}

impl Mlp {
    /// Inicializacion Xavier-uniforme; sesgos en cero.
    pub fn new(sizes: &[usize], acts: &[Activation], rng: &mut ChaCha8Rng) -> Mlp {
        assert_eq!(
            acts.len() + 1,
            sizes.len(),
            "Mlp::new: se necesitan {} activaciones para {} tamanos y se recibieron {}",
            sizes.len().saturating_sub(1),
            sizes.len(),
            acts.len()
        );
        let layers = sizes
            .windows(2)
            .zip(acts)
            .map(|(pair, &act)| {
                let (n_in, n_out) = (pair[0], pair[1]);
                let limit = (6.0 / (n_in + n_out) as f64).sqrt();
                let w = Array2::from_shape_fn((n_in, n_out), |_| rng.gen_range(-limit..=limit));
                DenseLayer { w, b: Array1::zeros(n_out), act }
            })
            .collect();
        Mlp { layers }
    }

    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut a = x.clone();
        for layer in &self.layers {
            a = layer.act.apply(&(a.dot(&layer.w) + &layer.b));
        }
        a
    }

    pub fn forward_cached(&self, x: &Array2<f64>) -> Cache {
        let mut a = Vec::with_capacity(self.layers.len() + 1);
        a.push(x.clone());
        for layer in &self.layers {
            let prev = a.last().expect("cache no vacia");
            let next = layer.act.apply(&(prev.dot(&layer.w) + &layer.b));
            a.push(next);
        }
        Cache { a }
    }

    /// `delta_last` es dL/dz (pre-activacion) de la ultima capa. Devuelve los gradientes de los
    /// parametros y dL/d(entrada) de la red.
    pub fn backward(&self, cache: &Cache, delta_last: Array2<f64>) -> (Grads, Array2<f64>) {
        let n = self.layers.len();
        let mut dw: Vec<Array2<f64>> = Vec::with_capacity(n);
        let mut db: Vec<Array1<f64>> = Vec::with_capacity(n);
        let mut delta = delta_last;
        let mut input_grad = Array2::zeros((0, 0));
        for l in (0..n).rev() {
            let layer = &self.layers[l];
            dw.push(cache.a[l].t().dot(&delta));
            db.push(delta.sum_axis(Axis(0)));
            let mut down = delta.dot(&layer.w.t());
            if l > 0 {
                down *= &self.layers[l - 1].act.derivative_from_output(&cache.a[l]);
                delta = down;
            } else {
                input_grad = down;
            }
        }
        dw.reverse();
        db.reverse();
        (Grads { dw, db }, input_grad)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Loss {
    Mse,
    /// Entropia cruzada binaria; requiere salida sigmoide.
    Bce,
}

impl Loss {
    /// Suma sobre las salidas de cada muestra, promedio sobre el batch.
    pub fn value(&self, pred: &Array2<f64>, target: &Array2<f64>) -> f64 {
        let batch = pred.nrows() as f64;
        match self {
            Loss::Mse => (pred - target).mapv(|d| d * d).sum() / (2.0 * batch),
            Loss::Bce => {
                const EPS: f64 = 1e-12;
                let total: f64 = pred
                    .iter()
                    .zip(target.iter())
                    .map(|(&p, &t)| {
                        let p = p.clamp(EPS, 1.0 - EPS);
                        t * p.ln() + (1.0 - t) * (1.0 - p).ln()
                    })
                    .sum();
                -total / batch
            }
        }
    }

    /// dL/dz de la capa de salida.
    pub fn output_delta(&self, pred: &Array2<f64>, target: &Array2<f64>, out_act: Activation) -> Array2<f64> {
        let batch = pred.nrows() as f64;
        match self {
            Loss::Mse => ((pred - target) / batch) * out_act.derivative_from_output(pred),
            // La derivada de la sigmoide se cancela analiticamente: dL/dz = (p - t) / batch.
            Loss::Bce => (pred - target) / batch,
        }
    }

    /// BCE solo es coherente con salida sigmoide (el delta asume la derivada cancelada).
    pub fn validate(&self, out_act: Activation) -> anyhow::Result<()> {
        if *self == Loss::Bce && out_act != Activation::Sigmoid {
            anyhow::bail!("la perdida BCE requiere activacion de salida Sigmoid, no {out_act:?}");
        }
        Ok(())
    }
}
