//! Evaluacion generativa del VAE (Ej 2, VAE-02 y VAE-03): juicio de las imagenes decodificadas contra
//! el dataset con umbrales derivados del propio dataset, busqueda de un emoji nuevo en el prior y
//! estadisticas de los sorteos del prior. Modulo numerico: no dibuja ni parsea argumentos.

use ndarray::{Array1, Array2, ArrayView1};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::emoji::{nearest_emoji, EmojiDataset, Separation};
use crate::vae::{standard_normal_matrix, Vae};

/// Un emoji es plausible si esta a lo sumo `PLAUSIBLE_FACTOR` veces el radio de cercania del mas cercano.
pub const PLAUSIBLE_FACTOR: usize = 2;
/// Un salto entre celdas vecinas es grande si supera `LARGE_STEP_FACTOR` veces el radio de cercania.
pub const LARGE_STEP_FACTOR: usize = 3;
/// Sorteos por defecto del prior.
pub const DEFAULT_DRAWS: usize = 1000;
/// Tope de `--draws`.
pub const MAX_DRAWS: usize = 100_000;
/// Cantidad maxima de muestras aceptadas que se guardan para la galeria.
pub const GALLERY_SIZE: usize = 8;
/// Filas por tanda al decodificar sorteos: la memoria no crece con `--draws`.
const DRAW_CHUNK: usize = 1000;

/// Valida el argumento `--draws`: entre 1 y `MAX_DRAWS`.
pub fn validate_draws(n: usize) -> anyhow::Result<usize> {
    if n == 0 || n > MAX_DRAWS {
        anyhow::bail!("el numero de sorteos debe estar entre 1 y {MAX_DRAWS} (se recibio {n})");
    }
    Ok(n)
}

/// Veredicto sobre una imagen decodificada, relativo al dataset.
#[derive(Debug, Clone, PartialEq)]
pub struct Judgement {
    pub nearest_index: usize,
    /// Pixeles (binarizando en 0.5) que la separan del emoji mas cercano.
    pub distance: usize,
    /// A lo sumo `near_radius` px del emoji mas cercano.
    pub recognizable: bool,
    /// Al menos `min` px de todos los emojis (tan lejos como los dos emojis mas cercanos entre si).
    pub is_new: bool,
    /// A lo sumo `PLAUSIBLE_FACTOR * near_radius` px del emoji mas cercano.
    pub plausible: bool,
}

impl Judgement {
    pub fn new_and_plausible(&self) -> bool {
        self.is_new && self.plausible
    }
}

pub fn judge(ds: &EmojiDataset, sep: &Separation, pattern: ArrayView1<f64>) -> Judgement {
    let (nearest_index, distance) = nearest_emoji(ds, pattern);
    Judgement {
        nearest_index,
        distance,
        recognizable: distance <= sep.near_radius,
        is_new: distance >= sep.min,
        plausible: distance <= PLAUSIBLE_FACTOR * sep.near_radius,
    }
}

#[derive(Debug, Clone)]
pub struct NewEmoji {
    /// Posicion (desde 0) del sorteo en el flujo semillado.
    pub draw_index: usize,
    pub z: Array1<f64>,
    /// Salida del decodificador (probabilidades en gris, sin binarizar).
    pub image: Array1<f64>,
    pub judgement: Judgement,
}

#[derive(Debug, Clone)]
pub struct SampleSearch {
    pub seed: u64,
    pub max_draws: usize,
    pub examined: usize,
    pub accepted: Vec<NewEmoji>,
}

/// Recorre en orden las filas del flujo semillado N(0, I) decodificadas por `decode`, en tandas de
/// `DRAW_CHUNK`; `visit(indice_global, z, imagen)` devuelve `false` para cortar. El orden de los
/// sorteos es el del flujo, asi que el resultado no depende del tamano de la tanda.
fn walk_prior(
    decode: &dyn Fn(&Array2<f64>) -> Array2<f64>,
    latent_dim: usize,
    seed: u64,
    n: usize,
    visit: &mut dyn FnMut(usize, ArrayView1<f64>, ArrayView1<f64>) -> bool,
) -> usize {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut done = 0usize;
    while done < n {
        let rows = DRAW_CHUNK.min(n - done);
        let z = standard_normal_matrix(&mut rng, rows, latent_dim);
        let images = decode(&z);
        for r in 0..rows {
            done += 1;
            if !visit(done - 1, z.row(r), images.row(r)) {
                return done;
            }
        }
    }
    done
}

/// Busca en el flujo semillado del prior los primeros sorteos que decodifican a un emoji nuevo y
/// plausible, hasta `GALLERY_SIZE` muestras o `max_draws` sorteos. El emoji principal es `accepted[0]`.
pub fn search_new_emoji(vae: &Vae, ds: &EmojiDataset, seed: u64, max_draws: usize) -> SampleSearch {
    let sep = crate::emoji::separation(ds);
    let mut accepted: Vec<NewEmoji> = Vec::new();
    let decode = |z: &Array2<f64>| vae.decode(z);
    let examined = walk_prior(&decode, vae.latent_dim, seed, max_draws, &mut |idx, z, img| {
        let j = judge(ds, &sep, img);
        if j.new_and_plausible() {
            accepted.push(NewEmoji { draw_index: idx, z: z.to_owned(), image: img.to_owned(), judgement: j });
        }
        accepted.len() < GALLERY_SIZE
    });
    SampleSearch { seed, max_draws, examined, accepted }
}

pub fn new_emoji_line(search: &SampleSearch, ds: &EmojiDataset, sep: &Separation) -> String {
    let criterion = format!(
        "criterio: a al menos {} px de todo el dataset y a lo sumo {} px del mas cercano",
        sep.min,
        PLAUSIBLE_FACTOR * sep.near_radius
    );
    match search.accepted.first() {
        Some(first) => format!(
            "EMOJI NUEVO: z=({:.4}, {:.4}) tras {} sorteos | emoji mas cercano: {} a {} px | {criterion}",
            first.z[0],
            first.z[1],
            first.draw_index + 1,
            ds.names[first.judgement.nearest_index],
            first.judgement.distance
        ),
        None => format!("EMOJI NUEVO: ninguno en {} sorteos ({criterion})", search.max_draws),
    }
}

/// Participacion de los sorteos del prior en cada categoria.
#[derive(Debug, Clone, PartialEq)]
pub struct PriorStats {
    pub n: usize,
    pub recognizable_frac: f64,
    pub new_frac: f64,
    pub new_plausible_frac: f64,
}

/// Estadisticas de los primeros `n` sorteos del flujo semillado, decodificados por `decode` (asi el
/// autoencoder simple reutiliza la misma funcion).
pub fn prior_stats(
    decode: &dyn Fn(&Array2<f64>) -> Array2<f64>,
    latent_dim: usize,
    ds: &EmojiDataset,
    sep: &Separation,
    seed: u64,
    n: usize,
) -> PriorStats {
    let (mut rec, mut new, mut both) = (0usize, 0usize, 0usize);
    let examined = walk_prior(decode, latent_dim, seed, n, &mut |_, _, img| {
        let j = judge(ds, sep, img);
        rec += usize::from(j.recognizable);
        new += usize::from(j.is_new);
        both += usize::from(j.new_and_plausible());
        true
    });
    let total = examined.max(1) as f64;
    PriorStats {
        n: examined,
        recognizable_frac: rec as f64 / total,
        new_frac: new as f64 / total,
        new_plausible_frac: both as f64 / total,
    }
}

pub fn prior_line(s: &PriorStats, sep: &Separation) -> String {
    format!(
        "PRIOR: de {} puntos z ~ N(0, I): {:.1} % decodifican a un emoji reconocible (a lo sumo {} px de un emoji del dataset), {:.1} % a un emoji nuevo (al menos {} px de todos) y {:.1} % a un emoji nuevo y plausible (a lo sumo {} px)",
        s.n,
        100.0 * s.recognizable_frac,
        sep.near_radius,
        100.0 * s.new_frac,
        sep.min,
        100.0 * s.new_plausible_frac,
        PLAUSIBLE_FACTOR * sep.near_radius
    )
}
