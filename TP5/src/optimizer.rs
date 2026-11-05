//! Optimizadores escritos a mano sobre ndarray: SGD, Momentum y Adam.

use ndarray::{Array1, Array2};

use crate::mlp::{Grads, Mlp};

pub trait Optimizer {
    fn step(&mut self, net: &mut Mlp, grads: &Grads);
}

/// Descenso por gradiente: `p = p - lr * g`.
#[derive(Debug, Clone)]
pub struct Sgd {
    pub lr: f64,
}

impl Optimizer for Sgd {
    fn step(&mut self, net: &mut Mlp, grads: &Grads) {
        for (l, layer) in net.layers.iter_mut().enumerate() {
            layer.w.scaled_add(-self.lr, &grads.dw[l]);
            layer.b.scaled_add(-self.lr, &grads.db[l]);
        }
    }
}

fn zero_state(net: &Mlp) -> (Vec<Array2<f64>>, Vec<Array1<f64>>) {
    (
        net.layers.iter().map(|l| Array2::zeros(l.w.raw_dim())).collect(),
        net.layers.iter().map(|l| Array1::zeros(l.b.raw_dim())).collect(),
    )
}

/// Momentum clasico (heavy-ball): `v = beta * v + g`, `p = p - lr * v`, con `v` inicial en cero.
#[derive(Debug, Clone)]
pub struct Momentum {
    pub lr: f64,
    pub beta: f64,
    vw: Vec<Array2<f64>>,
    vb: Vec<Array1<f64>>,
}

impl Momentum {
    pub fn new(lr: f64, beta: f64, net: &Mlp) -> Momentum {
        let (vw, vb) = zero_state(net);
        Momentum { lr, beta, vw, vb }
    }
}

impl Optimizer for Momentum {
    fn step(&mut self, net: &mut Mlp, grads: &Grads) {
        for (l, layer) in net.layers.iter_mut().enumerate() {
            self.vw[l] *= self.beta;
            self.vw[l] += &grads.dw[l];
            self.vb[l] *= self.beta;
            self.vb[l] += &grads.db[l];
            layer.w.scaled_add(-self.lr, &self.vw[l]);
            layer.b.scaled_add(-self.lr, &self.vb[l]);
        }
    }
}

/// Adam con correccion de sesgo de los momentos.
#[derive(Debug, Clone)]
pub struct Adam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    t: i32,
    mw: Vec<Array2<f64>>,
    mb: Vec<Array1<f64>>,
    vw: Vec<Array2<f64>>,
    vb: Vec<Array1<f64>>,
}

impl Adam {
    pub fn new(lr: f64, beta1: f64, beta2: f64, eps: f64, net: &Mlp) -> Adam {
        let (mw, mb) = zero_state(net);
        let (vw, vb) = zero_state(net);
        Adam { lr, beta1, beta2, eps, t: 0, mw, mb, vw, vb }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, net: &mut Mlp, grads: &Grads) {
        self.t += 1;
        let c1 = 1.0 - self.beta1.powi(self.t);
        let c2 = 1.0 - self.beta2.powi(self.t);
        let (b1, b2, lr, eps) = (self.beta1, self.beta2, self.lr, self.eps);
        for (l, layer) in net.layers.iter_mut().enumerate() {
            let g = &grads.dw[l];
            self.mw[l].zip_mut_with(g, |m, &g| *m = b1 * *m + (1.0 - b1) * g);
            self.vw[l].zip_mut_with(g, |v, &g| *v = b2 * *v + (1.0 - b2) * g * g);
            for ((w, &m), &v) in layer.w.iter_mut().zip(self.mw[l].iter()).zip(self.vw[l].iter()) {
                *w -= lr * (m / c1) / ((v / c2).sqrt() + eps);
            }
            let g = &grads.db[l];
            self.mb[l].zip_mut_with(g, |m, &g| *m = b1 * *m + (1.0 - b1) * g);
            self.vb[l].zip_mut_with(g, |v, &g| *v = b2 * *v + (1.0 - b2) * g * g);
            for ((b, &m), &v) in layer.b.iter_mut().zip(self.mb[l].iter()).zip(self.vb[l].iter()) {
                *b -= lr * (m / c1) / ((v / c2).sqrt() + eps);
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum OptimizerKind {
    Sgd { lr: f64 },
    Momentum { lr: f64, beta: f64 },
    Adam { lr: f64, beta1: f64, beta2: f64, eps: f64 },
}

impl OptimizerKind {
    /// Construye el optimizador con estado dimensionado segun `net`.
    pub fn build(&self, net: &Mlp) -> Box<dyn Optimizer> {
        match *self {
            OptimizerKind::Sgd { lr } => Box::new(Sgd { lr }),
            OptimizerKind::Momentum { lr, beta } => Box::new(Momentum::new(lr, beta, net)),
            OptimizerKind::Adam { lr, beta1, beta2, eps } => Box::new(Adam::new(lr, beta1, beta2, eps, net)),
        }
    }

    pub fn label(&self) -> String {
        match *self {
            OptimizerKind::Sgd { lr } => format!("SGD(lr={lr})"),
            OptimizerKind::Momentum { lr, beta } => format!("Momentum(lr={lr},beta={beta})"),
            OptimizerKind::Adam { lr, .. } => format!("Adam(lr={lr})"),
        }
    }
}
