//! Modelo de ruido del Denoising Autoencoder (Ej 1b): inversion independiente de pixeles.
//!
//! El nivel `p` de ruido es la probabilidad de que cada pixel se invierta (`v -> 1 - v`), la
//! version para imagenes binarias del ruido sal y pimienta de la teoria. Este modulo no dibuja ni
//! parsea la linea de comandos.

use anyhow::{bail, Result};
use ndarray::{concatenate, Array2, ArrayView1, ArrayViewMut1, Axis};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Nivel maximo aceptado: con 0.5 la entrada no lleva informacion del patron.
pub const MAX_LEVEL: f64 = 0.5;

/// Cantidad maxima de niveles que acepta la linea de comandos.
pub const MAX_LEVELS: usize = 12;

/// Un nivel valido es finito y esta en 0.0 ..= `MAX_LEVEL`.
pub fn validate_level(level: f64) -> Result<()> {
    if !level.is_finite() {
        bail!("nivel de ruido invalido: {level} (debe ser un numero finito entre 0.0 y {MAX_LEVEL})");
    }
    if !(0.0..=MAX_LEVEL).contains(&level) {
        bail!("nivel de ruido fuera de rango: {level} (debe estar entre 0.0 y {MAX_LEVEL})");
    }
    Ok(())
}

/// Valida una lista de niveles (no vacia, a lo sumo `MAX_LEVELS`, cada uno valido, sin repetidos)
/// y la devuelve ordenada de menor a mayor.
pub fn validate_levels(levels: &[f64]) -> Result<Vec<f64>> {
    if levels.is_empty() {
        bail!("la lista de niveles de ruido esta vacia (se necesita al menos un nivel)");
    }
    if levels.len() > MAX_LEVELS {
        bail!("demasiados niveles de ruido: {} (el maximo es {MAX_LEVELS})", levels.len());
    }
    for &l in levels {
        validate_level(l)?;
    }
    let mut sorted = levels.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    if let Some(w) = sorted.windows(2).find(|w| w[0] == w[1]) {
        bail!("nivel de ruido repetido: {} (cada nivel debe aparecer una sola vez)", w[0]);
    }
    Ok(sorted)
}

/// Generador determinista para un par (semilla, nivel): no depende de que otros niveles se pidan.
pub fn level_rng(seed: u64, level: f64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed ^ level.to_bits())
}

/// Copia `src` en `dst` invirtiendo cada pixel cuando el sorteo cae por debajo de `level`.
/// Consume exactamente un `f64` del generador por elemento.
fn flip_row(src: ArrayView1<f64>, mut dst: ArrayViewMut1<f64>, level: f64, rng: &mut ChaCha8Rng) {
    for (d, &v) in dst.iter_mut().zip(src.iter()) {
        *d = if rng.gen::<f64>() < level { 1.0 - v } else { v };
    }
}

/// Devuelve una copia de `x` con cada pixel invertido con probabilidad `level`, recorriendo en
/// orden fila a fila. Con nivel 0.0 la copia es exacta. Entra en panico con un nivel invalido
/// (la linea de comandos valida antes de llamar).
pub fn distort(x: &Array2<f64>, level: f64, rng: &mut ChaCha8Rng) -> Array2<f64> {
    if let Err(e) = validate_level(level) {
        panic!("{e}");
    }
    let mut out = Array2::zeros(x.raw_dim());
    for (src, dst) in x.rows().into_iter().zip(out.rows_mut()) {
        flip_row(src, dst, level, rng);
    }
    out
}

/// Nivel de ruido usado al entrenar: fijo o sorteado por patron en un rango.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoiseSchedule {
    Fixed(f64),
    Uniform { lo: f64, hi: f64 },
}

impl NoiseSchedule {
    pub fn validate(&self) -> Result<()> {
        match *self {
            NoiseSchedule::Fixed(p) => validate_level(p),
            NoiseSchedule::Uniform { lo, hi } => {
                validate_level(lo)?;
                validate_level(hi)?;
                if lo > hi {
                    bail!("rango de ruido invalido: el minimo {lo} supera al maximo {hi}");
                }
                Ok(())
            }
        }
    }

    /// `fix0.10` o `u0.00-0.20`.
    pub fn label(&self) -> String {
        match *self {
            NoiseSchedule::Fixed(p) => format!("fix{p:.2}"),
            NoiseSchedule::Uniform { lo, hi } => format!("u{lo:.2}-{hi:.2}"),
        }
    }

    /// `Fixed` no consume el generador; `Uniform` sortea un nivel en `lo..=hi`.
    pub fn sample_level(&self, rng: &mut ChaCha8Rng) -> f64 {
        match *self {
            NoiseSchedule::Fixed(p) => p,
            NoiseSchedule::Uniform { lo, hi } => rng.gen_range(lo..=hi),
        }
    }
}

/// Flujo de ruido de entrenamiento: su propio generador, un nivel sorteado por fila.
pub struct TrainNoiseStream {
    schedule: NoiseSchedule,
    rng: ChaCha8Rng,
}

impl TrainNoiseStream {
    pub fn new(schedule: NoiseSchedule, seed: u64) -> TrainNoiseStream {
        if let Err(e) = schedule.validate() {
            panic!("{e}");
        }
        TrainNoiseStream { schedule, rng: ChaCha8Rng::seed_from_u64(seed) }
    }

    /// Una version ruidosa nueva de `clean`: para cada fila se sortea un nivel del cronograma y
    /// luego se invierte cada pixel de esa fila con ese nivel.
    pub fn next_batch(&mut self, clean: &Array2<f64>) -> Array2<f64> {
        let mut out = Array2::zeros(clean.raw_dim());
        for (src, dst) in clean.rows().into_iter().zip(out.rows_mut()) {
            let level = self.schedule.sample_level(&mut self.rng);
            flip_row(src, dst, level, &mut self.rng);
        }
        out
    }
}

/// Repite la matriz completa `copies` veces a lo largo de las filas.
pub fn tile_rows(x: &Array2<f64>, copies: usize) -> Array2<f64> {
    assert!(copies >= 1, "copies debe ser al menos 1");
    let views = vec![x.view(); copies];
    concatenate(Axis(0), &views).expect("las copias tienen las mismas columnas")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::load_font;

    #[test]
    fn level_zero_is_identity() {
        let ds = load_font().unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let y = distort(&ds.x, 0.0, &mut rng);
        assert_eq!(y, ds.x);
        assert!(y.iter().zip(ds.x.iter()).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    #[test]
    fn flip_rate_matches_level() {
        let ds = load_font().unwrap();
        let big = tile_rows(&ds.x, 50);
        for level in [0.05, 0.1, 0.5] {
            let mut rng = level_rng(99, level);
            let y = distort(&big, level, &mut rng);
            let flipped = y.iter().zip(big.iter()).filter(|(a, b)| a != b).count();
            let rate = flipped as f64 / big.len() as f64;
            assert!((rate - level).abs() < 0.01, "nivel {level}: tasa medida {rate}");
        }
    }

    #[test]
    fn distort_is_binary_and_leaves_input_untouched() {
        let ds = load_font().unwrap();
        let before = ds.x.clone();
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let y = distort(&ds.x, 0.3, &mut rng);
        assert_eq!(ds.x, before);
        assert!(y.iter().all(|&v| v == 0.0 || v == 1.0));
        assert_ne!(y, ds.x);
    }

    #[test]
    fn distort_is_deterministic_by_seed() {
        let ds = load_font().unwrap();
        let a = distort(&ds.x, 0.2, &mut ChaCha8Rng::seed_from_u64(5));
        let b = distort(&ds.x, 0.2, &mut ChaCha8Rng::seed_from_u64(5));
        let c = distort(&ds.x, 0.2, &mut ChaCha8Rng::seed_from_u64(6));
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn validate_level_bounds() {
        assert!(validate_level(0.0).is_ok());
        assert!(validate_level(0.5).is_ok());
        assert!(validate_level(-0.01).is_err());
        assert!(validate_level(0.5000001).is_err());
        assert!(validate_level(f64::NAN).is_err());
        assert!(validate_level(f64::INFINITY).is_err());
        assert!(validate_level(f64::NEG_INFINITY).is_err());
    }

    #[test]
    fn uniform_schedule_samples_inside_its_range() {
        let s = NoiseSchedule::Uniform { lo: 0.1, hi: 0.3 };
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        for _ in 0..1000 {
            let p = s.sample_level(&mut rng);
            assert!((0.1..=0.3).contains(&p), "nivel sorteado {p}");
        }
        assert_eq!(s.label(), "u0.10-0.30");
        assert_eq!(NoiseSchedule::Fixed(0.1).label(), "fix0.10");
        assert!(NoiseSchedule::Uniform { lo: 0.3, hi: 0.1 }.validate().is_err());
    }

    #[test]
    fn stream_with_fixed_zero_never_flips() {
        let ds = load_font().unwrap();
        let mut stream = TrainNoiseStream::new(NoiseSchedule::Fixed(0.0), 4);
        for _ in 0..5 {
            assert_eq!(stream.next_batch(&ds.x), ds.x);
        }
    }
}
