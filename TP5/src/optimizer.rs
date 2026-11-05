//! Optimizadores escritos a mano sobre ndarray.

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

#[derive(Debug, Clone, PartialEq)]
pub enum OptimizerKind {
    Sgd { lr: f64 },
}

impl OptimizerKind {
    /// Construye el optimizador con estado dimensionado segun `net`.
    pub fn build(&self, _net: &Mlp) -> Box<dyn Optimizer> {
        match *self {
            OptimizerKind::Sgd { lr } => Box::new(Sgd { lr }),
        }
    }

    pub fn label(&self) -> String {
        match *self {
            OptimizerKind::Sgd { lr } => format!("SGD(lr={lr})"),
        }
    }
}
