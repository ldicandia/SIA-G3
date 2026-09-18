pub mod config;
pub mod data;
pub mod digits;
pub mod exercise2;
pub mod exercise3;
pub mod experiment;
pub mod loss;
pub mod matrix;
pub mod metrics;
pub mod model;
pub mod split;
pub mod training;

pub use loss::{categorical_cross_entropy, Loss, MeanSquaredError};
pub use matrix::DenseMatrix;
pub use model::{Activation, MultilayerPerceptron, SingleLayerPerceptron, StepPerceptron};
