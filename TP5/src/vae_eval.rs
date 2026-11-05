//! Evaluacion generativa del VAE (Ej 2, VAE-02 y VAE-03): juicio de las imagenes decodificadas contra
//! el dataset con umbrales derivados del propio dataset, busqueda de un emoji nuevo en el prior y
//! estadisticas de los sorteos del prior. Modulo numerico: no dibuja ni parsea argumentos.

use std::collections::{HashMap, HashSet};

use ndarray::{Array1, Array2, ArrayView1};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::autoencoder::{train, Autoencoder, TrainConfig};
use crate::emoji::{
    hamming, nearest_emoji, region_hamming, separation, style_regions, EmojiDataset, Separation, StyleRegions,
    EMOJI_EYES, EMOJI_MOUTHS, EMOJI_PIXELS,
};
use crate::report::TOLERANCE_PX;
use crate::vae::{reparameterize, standard_normal_matrix, ReconSummary, Vae, VaeConfig, VaeHistory};

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
}

/// Emoji global mas cercano sobre los 576 pixeles; lo usan el conteo de reconocibles, las metricas de la
/// grilla y las figuras. Que una imagen sea "nueva" lo decide `StyleJudge`, region por region.
pub fn judge(ds: &EmojiDataset, sep: &Separation, pattern: ArrayView1<f64>) -> Judgement {
    let (nearest_index, distance) = nearest_emoji(ds, pattern);
    Judgement {
        nearest_index,
        distance,
        recognizable: distance <= sep.near_radius,
    }
}

/// Una region fuera de estilo solo es plausible si queda entre sus dos estilos mas cercanos: distancia al
/// mas cercano mas distancia al segundo, a lo sumo este factor por la distancia entre esos dos estilos.
pub const BLEND_SLACK: f64 = 1.5;

/// Piso de ruido de reconstruccion por region: el mayor error que el VAE comete en esa region al
/// reconstruir los 36 emojis de entrenamiento desde la media posterior.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseFloors {
    pub eyes: usize,
    pub mouth: usize,
    pub frame: usize,
}

/// `recon` tiene en la fila `i` la reconstruccion del emoji de entrenamiento `i`.
pub fn noise_floors(ds: &EmojiDataset, regions: &StyleRegions, recon: &Array2<f64>) -> NoiseFloors {
    let worst = |pixels: &[usize]| {
        (0..ds.x.nrows()).map(|i| region_hamming(recon.row(i), ds.x.row(i), pixels)).max().unwrap_or(0)
    };
    NoiseFloors { eyes: worst(&regions.eyes), mouth: worst(&regions.mouth), frame: worst(&regions.frame) }
}

/// Veredicto de una region (ojos o boca) frente a sus 6 estilos.
#[derive(Debug, Clone, PartialEq)]
pub struct RegionVerdict {
    /// Estilo mas cercano de la region (ante empates, el de menor indice).
    pub style: usize,
    pub distance: usize,
    /// Segundo estilo mas cercano (ante empates, el de menor indice entre el resto).
    pub second_style: usize,
    pub second_distance: usize,
    /// Mas lejos que el piso de ruido de los 6 estilos de la region.
    pub off_style: bool,
    /// La region no es fuera de estilo, o queda entre sus dos estilos mas cercanos (`BLEND_SLACK`).
    pub blend_ok: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StyleVerdict {
    pub eyes: RegionVerdict,
    pub mouth: RegionVerdict,
    /// Distancia Hamming sobre los pixeles del marco a la fila 0 del dataset.
    pub frame_distance: usize,
    pub plausible: bool,
    /// "Nuevo": alguna region fuera de estilo y la cara plausible.
    pub hybrid: bool,
    /// Mayor `distancia - piso` entre las regiones fuera de estilo (0 si ninguna lo esta).
    pub margin: usize,
}

/// Juez de novedad por region. Los prototipos son filas del dataset: el estilo de ojos `e` es la fila
/// `e * 6` y el de boca `m` es la fila `m` (cada region depende solo de su propio estilo).
#[derive(Debug, Clone)]
pub struct StyleJudge {
    ds: EmojiDataset,
    sep: Separation,
    regions: StyleRegions,
    floors: NoiseFloors,
    eyes_gap: [[usize; 6]; 6],
    mouth_gap: [[usize; 6]; 6],
}

fn eyes_prototype(style: usize) -> usize {
    style * EMOJI_MOUTHS.len()
}

fn mouth_prototype(style: usize) -> usize {
    style
}

fn prototype_gaps(ds: &EmojiDataset, pixels: &[usize], row_of: fn(usize) -> usize) -> [[usize; 6]; 6] {
    let mut gap = [[0usize; 6]; 6];
    for a in 0..6 {
        for b in 0..6 {
            gap[a][b] = region_hamming(ds.x.row(row_of(a)), ds.x.row(row_of(b)), pixels);
        }
    }
    gap
}

impl StyleJudge {
    /// Pisos medidos sobre `recon`, las reconstrucciones del modelo de los 36 emojis de entrenamiento.
    pub fn new(ds: &EmojiDataset, recon: &Array2<f64>) -> StyleJudge {
        let floors = noise_floors(ds, &style_regions(ds), recon);
        StyleJudge::with_floors(ds, floors)
    }

    pub fn with_floors(ds: &EmojiDataset, floors: NoiseFloors) -> StyleJudge {
        let regions = style_regions(ds);
        let eyes_gap = prototype_gaps(ds, &regions.eyes, eyes_prototype);
        let mouth_gap = prototype_gaps(ds, &regions.mouth, mouth_prototype);
        StyleJudge { ds: ds.clone(), sep: separation(ds), regions, floors, eyes_gap, mouth_gap }
    }

    pub fn floors(&self) -> NoiseFloors {
        self.floors
    }

    pub fn regions(&self) -> &StyleRegions {
        &self.regions
    }

    pub fn dataset(&self) -> &EmojiDataset {
        &self.ds
    }

    pub fn separation(&self) -> &Separation {
        &self.sep
    }

    /// Menor distancia entre dos prototipos de estilos distintos dentro de la region de ojos y de boca.
    pub fn min_style_separation(&self) -> (usize, usize) {
        let min_gap = |gap: &[[usize; 6]; 6]| {
            (0..6).flat_map(|a| ((a + 1)..6).map(move |b| gap[a][b])).min().expect("hay mas de un estilo")
        };
        (min_gap(&self.eyes_gap), min_gap(&self.mouth_gap))
    }

    fn region_verdict(
        &self,
        pattern: ArrayView1<f64>,
        pixels: &[usize],
        floor: usize,
        row_of: fn(usize) -> usize,
        gap: &[[usize; 6]; 6],
    ) -> RegionVerdict {
        let dists: Vec<usize> =
            (0..6).map(|s| region_hamming(pattern, self.ds.x.row(row_of(s)), pixels)).collect();
        let mut style = 0usize;
        for s in 1..6 {
            if dists[s] < dists[style] {
                style = s;
            }
        }
        let mut second_style = usize::MAX;
        for s in 0..6 {
            if s != style && (second_style == usize::MAX || dists[s] < dists[second_style]) {
                second_style = s;
            }
        }
        let (distance, second_distance) = (dists[style], dists[second_style]);
        let off_style = distance > floor;
        let between = (distance + second_distance) as f64 <= BLEND_SLACK * gap[style][second_style] as f64;
        RegionVerdict { style, distance, second_style, second_distance, off_style, blend_ok: !off_style || between }
    }

    pub fn judge(&self, pattern: ArrayView1<f64>) -> StyleVerdict {
        let eyes =
            self.region_verdict(pattern, &self.regions.eyes, self.floors.eyes, eyes_prototype, &self.eyes_gap);
        let mouth =
            self.region_verdict(pattern, &self.regions.mouth, self.floors.mouth, mouth_prototype, &self.mouth_gap);
        let frame_distance = region_hamming(pattern, self.ds.x.row(0), &self.regions.frame);
        let plausible = frame_distance <= self.floors.frame && eyes.blend_ok && mouth.blend_ok;
        let hybrid = (eyes.off_style || mouth.off_style) && plausible;
        let excess = |r: &RegionVerdict, floor: usize| if r.off_style { r.distance - floor } else { 0 };
        let margin = excess(&eyes, self.floors.eyes).max(excess(&mouth, self.floors.mouth));
        StyleVerdict { eyes, mouth, frame_distance, plausible, hybrid, margin }
    }
}

/// Linea `REGIONES:` con el tamano de cada region, los pisos de ruido y la separacion entre estilos.
pub fn regions_line(judge: &StyleJudge) -> String {
    let (r, f) = (judge.regions(), judge.floors());
    let (se, sm) = judge.min_style_separation();
    format!(
        "REGIONES: ojos {} px, boca {} px, marco {} px | piso de ruido de reconstruccion (maximo sobre los 36 emojis): ojos {} px, boca {} px, marco {} px | separacion minima entre estilos: ojos {} px, boca {} px",
        r.eyes.len(),
        r.mouth.len(),
        r.frame.len(),
        f.eyes,
        f.mouth,
        f.frame,
        se,
        sm
    )
}

#[derive(Debug, Clone)]
pub struct NewEmoji {
    /// Posicion (desde 0) del sorteo en el flujo semillado.
    pub draw_index: usize,
    pub z: Array1<f64>,
    /// Salida del decodificador (probabilidades en gris, sin binarizar).
    pub image: Array1<f64>,
    /// Emoji del dataset mas cercano sobre los 576 pixeles.
    pub judgement: Judgement,
    /// Veredicto por region (ojos / boca / marco).
    pub verdict: StyleVerdict,
}

#[derive(Debug, Clone)]
pub struct SampleSearch {
    pub seed: u64,
    pub max_draws: usize,
    pub examined: usize,
    /// Pisos de ruido de reconstruccion del VAE usados para juzgar.
    pub floors: NoiseFloors,
    /// Sorteos que resultaron hibridos (nuevos y plausibles).
    pub candidates: usize,
    /// Combinaciones distintas (emoji mas cercano, estilo de ojos, estilo de boca) entre los hibridos.
    pub distinct_keys: usize,
    /// A lo sumo `GALLERY_SIZE`, una por combinacion, ordenadas por margen descendente y luego por sorteo.
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

/// Recorre los `max_draws` sorteos del flujo semillado (sin corte anticipado: el ranking necesita todo el
/// recorrido) y juzga cada imagen decodificada por region. De los sorteos hibridos conserva, por cada
/// combinacion (emoji mas cercano, estilo de ojos, estilo de boca), el de mayor margen (ante empates, el
/// sorteo anterior). La galeria son las `GALLERY_SIZE` mejores combinaciones por margen descendente y luego
/// por sorteo; el emoji principal es `accepted[0]`. La memoria crece con las combinaciones, no con
/// `max_draws`, y el resultado no depende del tamano de la tanda.
pub fn search_new_emoji(vae: &Vae, ds: &EmojiDataset, seed: u64, max_draws: usize) -> SampleSearch {
    let style_judge = StyleJudge::new(ds, &vae.reconstruct_mean(&ds.x));
    let sep = style_judge.separation().clone();
    let mut best: HashMap<(usize, usize, usize), NewEmoji> = HashMap::new();
    let mut candidates = 0usize;
    let decode = |z: &Array2<f64>| vae.decode(z);
    let examined = walk_prior(&decode, vae.latent_dim, seed, max_draws, &mut |idx, z, img| {
        let verdict = style_judge.judge(img);
        if verdict.hybrid {
            candidates += 1;
            let judgement = judge(ds, &sep, img);
            let key = (judgement.nearest_index, verdict.eyes.style, verdict.mouth.style);
            let better = best.get(&key).map_or(true, |current| verdict.margin > current.verdict.margin);
            if better {
                best.insert(
                    key,
                    NewEmoji { draw_index: idx, z: z.to_owned(), image: img.to_owned(), judgement, verdict },
                );
            }
        }
        true
    });
    let distinct_keys = best.len();
    let mut accepted: Vec<NewEmoji> = best.into_values().collect();
    accepted.sort_by(|a, b| b.verdict.margin.cmp(&a.verdict.margin).then(a.draw_index.cmp(&b.draw_index)));
    accepted.truncate(GALLERY_SIZE);
    SampleSearch { seed, max_draws, examined, floors: style_judge.floors(), candidates, distinct_keys, accepted }
}

const NEW_CRITERION: &str = "alguna region a mas de su piso de ruido de todos los estilos de esa region, con el marco intacto y cada region en un estilo o entre dos";

pub fn new_emoji_line(search: &SampleSearch, ds: &EmojiDataset, sep: &Separation) -> String {
    match search.accepted.first() {
        Some(h) => format!(
            "EMOJI NUEVO: z=({:.4}, {:.4}) sorteo {} de {} | emoji mas cercano: {} a {} px (radio de cercania {} px) | ojos: {} a {} px del estilo mas cercano (piso {} px), boca: {} a {} px (piso {} px) | criterio: {NEW_CRITERION}",
            h.z[0],
            h.z[1],
            h.draw_index + 1,
            search.examined,
            ds.names[h.judgement.nearest_index],
            h.judgement.distance,
            sep.near_radius,
            EMOJI_EYES[h.verdict.eyes.style],
            h.verdict.eyes.distance,
            search.floors.eyes,
            EMOJI_MOUTHS[h.verdict.mouth.style],
            h.verdict.mouth.distance,
            search.floors.mouth
        ),
        None => format!("EMOJI NUEVO: ninguno en {} sorteos (criterio: {NEW_CRITERION})", search.max_draws),
    }
}

/// Galeria de las muestras aceptadas: una linea de encabezado y una por muestra, con dos espacios al inicio.
pub fn render_gallery(search: &SampleSearch, ds: &EmojiDataset) -> String {
    let mut lines = vec![format!(
        "GALERIA: {} sorteos candidatos de {}, {} combinaciones distintas (emoji mas cercano, estilo de ojos, estilo de boca); se muestran {} ordenadas por margen sobre el piso de ruido",
        search.candidates,
        search.examined,
        search.distinct_keys,
        search.accepted.len()
    )];
    for (i, s) in search.accepted.iter().enumerate() {
        lines.push(format!(
            "  nuevo {}: sorteo {} | emoji mas cercano {} a {} px | ojos {} a {} px | boca {} a {} px | margen {} px",
            i + 1,
            s.draw_index + 1,
            ds.names[s.judgement.nearest_index],
            s.judgement.distance,
            EMOJI_EYES[s.verdict.eyes.style],
            s.verdict.eyes.distance,
            EMOJI_MOUTHS[s.verdict.mouth.style],
            s.verdict.mouth.distance,
            s.verdict.margin
        ));
    }
    lines.join("\n")
}

/// Participacion de los sorteos del prior en cada categoria.
#[derive(Debug, Clone, PartialEq)]
pub struct PriorStats {
    pub n: usize,
    /// A lo sumo `near_radius` px (distancia global) de un emoji del dataset.
    pub recognizable_frac: f64,
    /// Cara valida: marco dentro de su piso y cada region en un estilo o entre dos.
    pub plausible_frac: f64,
    /// Alguna region fuera de estilo (plausible o no).
    pub off_style_frac: f64,
    /// Alguna region fuera de estilo y la cara plausible: el predicado que acepta un sorteo como nuevo.
    pub hybrid_frac: f64,
    /// Combinaciones distintas (emoji mas cercano, estilo de ojos, estilo de boca) entre los hibridos.
    pub hybrid_keys: usize,
}

/// Estadisticas de los primeros `n` sorteos del flujo semillado, decodificados por `decode` (asi el
/// autoencoder simple reutiliza la misma funcion con su propio juez).
pub fn prior_stats(
    decode: &dyn Fn(&Array2<f64>) -> Array2<f64>,
    latent_dim: usize,
    style_judge: &StyleJudge,
    seed: u64,
    n: usize,
) -> PriorStats {
    let (ds, sep) = (style_judge.dataset(), style_judge.separation());
    let (mut rec, mut plausible, mut off_style, mut hybrid) = (0usize, 0usize, 0usize, 0usize);
    let mut keys: HashSet<(usize, usize, usize)> = HashSet::new();
    let examined = walk_prior(decode, latent_dim, seed, n, &mut |_, _, img| {
        let global = judge(ds, sep, img);
        rec += usize::from(global.recognizable);
        let verdict = style_judge.judge(img);
        plausible += usize::from(verdict.plausible);
        off_style += usize::from(verdict.eyes.off_style || verdict.mouth.off_style);
        if verdict.hybrid {
            hybrid += 1;
            keys.insert((global.nearest_index, verdict.eyes.style, verdict.mouth.style));
        }
        true
    });
    let total = examined.max(1) as f64;
    PriorStats {
        n: examined,
        recognizable_frac: rec as f64 / total,
        plausible_frac: plausible as f64 / total,
        off_style_frac: off_style as f64 / total,
        hybrid_frac: hybrid as f64 / total,
        hybrid_keys: keys.len(),
    }
}

pub fn prior_line(s: &PriorStats, sep: &Separation) -> String {
    format!(
        "PRIOR: de {} puntos z ~ N(0, I): {:.1} % decodifican a un emoji reconocible (a lo sumo {} px de un emoji del dataset), {:.1} % a una cara plausible (marco intacto y cada region en un estilo o entre dos), {:.1} % a un hibrido plausible (alguna region fuera del piso de ruido de todos los estilos) en {} combinaciones distintas",
        s.n,
        100.0 * s.recognizable_frac,
        sep.near_radius,
        100.0 * s.plausible_frac,
        100.0 * s.hybrid_frac,
        s.hybrid_keys
    )
}

/// Lado por defecto de la grilla del plano latente.
pub const GRID_SIDE_DEFAULT: usize = 15;
pub const GRID_SIDE_MIN: usize = 3;
pub const GRID_SIDE_MAX: usize = 41;
/// La grilla cubre `[-GRID_HALF_RANGE, GRID_HALF_RANGE]` en cada eje.
pub const GRID_HALF_RANGE: f64 = 2.5;

/// Valida el argumento `--grid`: entre `GRID_SIDE_MIN` y `GRID_SIDE_MAX`.
pub fn validate_grid(side: usize) -> anyhow::Result<usize> {
    if !(GRID_SIDE_MIN..=GRID_SIDE_MAX).contains(&side) {
        anyhow::bail!("el lado de la grilla debe estar entre {GRID_SIDE_MIN} y {GRID_SIDE_MAX} (se recibio {side})");
    }
    Ok(side)
}

/// `side * side` puntos de 2 columnas en orden fila mayor de celdas: z1 crece hacia la derecha y z2
/// decrece hacia abajo (el plano se lee como un grafico con z2 hacia arriba).
pub fn latent_grid(side: usize, half_range: f64) -> Array2<f64> {
    let denom = (side - 1) as f64;
    let mut g = Array2::zeros((side * side, 2));
    for r in 0..side {
        for c in 0..side {
            g[[r * side + c, 0]] = -half_range + 2.0 * half_range * c as f64 / denom;
            g[[r * side + c, 1]] = half_range - 2.0 * half_range * r as f64 / denom;
        }
    }
    g
}

/// Decodifica toda la grilla en un solo lote (el VAE debe tener latente de 2 dimensiones).
pub fn decode_grid(vae: &Vae, side: usize, half_range: f64) -> Array2<f64> {
    assert!(
        vae.latent_dim == 2,
        "la grilla del plano latente necesita un VAE con latente de 2 dimensiones (tiene {})",
        vae.latent_dim
    );
    vae.decode(&latent_grid(side, half_range))
}

#[derive(Debug, Clone, PartialEq)]
pub struct GridMetrics {
    pub side: usize,
    pub n_cells: usize,
    /// Participacion de celdas a lo sumo `near_radius` px de un emoji del dataset.
    pub recognizable_frac: f64,
    /// Media de la distancia Hamming entre celdas vecinas (horizontales y verticales).
    pub mean_adjacent_step: f64,
    pub max_adjacent_step: usize,
    /// Participacion de pares vecinos con un salto de mas de `LARGE_STEP_FACTOR * near_radius` px.
    pub large_step_frac: f64,
}

pub fn manifold_metrics(decoded: &Array2<f64>, side: usize, ds: &EmojiDataset, sep: &Separation) -> GridMetrics {
    assert_eq!(decoded.nrows(), side * side, "la grilla decodificada debe tener side * side filas");
    let n_cells = side * side;
    let recognizable = decoded.rows().into_iter().filter(|r| nearest_emoji(ds, *r).1 <= sep.near_radius).count();
    let mut steps = Vec::with_capacity(2 * side * (side - 1));
    for r in 0..side {
        for c in 0..side {
            let here = decoded.row(r * side + c);
            if c + 1 < side {
                steps.push(hamming(here, decoded.row(r * side + c + 1)));
            }
            if r + 1 < side {
                steps.push(hamming(here, decoded.row((r + 1) * side + c)));
            }
        }
    }
    let large = steps.iter().filter(|&&s| s > LARGE_STEP_FACTOR * sep.near_radius).count();
    GridMetrics {
        side,
        n_cells,
        recognizable_frac: recognizable as f64 / n_cells as f64,
        mean_adjacent_step: steps.iter().sum::<usize>() as f64 / steps.len() as f64,
        max_adjacent_step: steps.iter().copied().max().unwrap_or(0),
        large_step_frac: large as f64 / steps.len() as f64,
    }
}

pub fn manifold_line(m: &GridMetrics, sep: &Separation) -> String {
    format!(
        "MANIFOLD: {} celdas ({}x{}) sobre z en [-{:.1}, {:.1}]^2 | reconocibles {:.1} % (a lo sumo {} px de un emoji del dataset) | salto medio entre celdas vecinas {:.1} px (maximo {}) | saltos de mas de {} px: {:.1} %",
        m.n_cells,
        m.side,
        m.side,
        GRID_HALF_RANGE,
        GRID_HALF_RANGE,
        100.0 * m.recognizable_frac,
        sep.near_radius,
        m.mean_adjacent_step,
        m.max_adjacent_step,
        LARGE_STEP_FACTOR * sep.near_radius,
        100.0 * m.large_step_frac
    )
}

// Umbrales de aceptacion: convenciones de este trabajo calibradas con margen, documentadas y no
// ajustadas para aprobar.
pub const MAX_MEAN_WRONG_PX: f64 = 12.0;
pub const MIN_KL: f64 = 1.0;
pub const MIN_GRID_RECOGNIZABLE: f64 = 0.5;
pub const MAX_ADJACENT_FACTOR: f64 = 1.5;
pub const MIN_PRIOR_RECOGNIZABLE: f64 = 0.6;
/// Participacion minima de sorteos hibridos y plausibles. Reemplaza la tasa de banda de pixeles (0.10), que
/// el autoencoder simple tambien cumplia (31.9 % contra 30.7 %) y por eso no distinguia los modelos.
pub const MIN_NEW_RATE: f64 = 0.05;
pub const MIN_CONTRAST_GAP: f64 = 0.3;
/// Ventaja minima de sorteos plausibles del VAE sobre el autoencoder simple (calibrada: unos 0.64).
pub const MIN_VALID_GAP: f64 = 0.3;

/// Autoencoder simple con la misma arquitectura, optimizador, perdida, epocas y semilla que el VAE,
/// pero sin capa estocastica ni termino KL.
pub fn train_plain_ae(ds: &EmojiDataset, cfg: &VaeConfig) -> Autoencoder {
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    let mut ae = Autoencoder::new(&cfg.spec, EMOJI_PIXELS, &mut rng);
    let tcfg = TrainConfig { epochs: cfg.epochs, eval_every: cfg.epochs, tolerance_px: TOLERANCE_PX, early_stop: false };
    train(&mut ae, &ds.x, &ds.x, cfg.recon, &cfg.optimizer, &tcfg);
    ae
}

/// Puntos de una grilla `side x side` sobre la caja de los codigos de entrenamiento del autoencoder
/// simple, con 10 % del rango de cada eje de margen a cada lado, en el orden y la orientacion de
/// `latent_grid` (z1 crece hacia la derecha, z2 decrece hacia abajo).
pub fn plain_grid_points(ae: &Autoencoder, ds: &EmojiDataset, side: usize) -> Array2<f64> {
    let codes = ae.encode(&ds.x);
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for row in codes.rows() {
        for a in 0..2 {
            lo[a] = lo[a].min(row[a]);
            hi[a] = hi[a].max(row[a]);
        }
    }
    let pad = [0.1 * (hi[0] - lo[0]), 0.1 * (hi[1] - lo[1])];
    let (x0, x1) = (lo[0] - pad[0], hi[0] + pad[0]);
    let (y0, y1) = (lo[1] - pad[1], hi[1] + pad[1]);
    let denom = (side - 1) as f64;
    let mut g = Array2::zeros((side * side, 2));
    for r in 0..side {
        for c in 0..side {
            g[[r * side + c, 0]] = x0 + (x1 - x0) * c as f64 / denom;
            g[[r * side + c, 1]] = y1 - (y1 - y0) * r as f64 / denom;
        }
    }
    g
}

#[derive(Debug, Clone)]
pub struct Contrast {
    pub vae_prior: PriorStats,
    pub plain_prior: PriorStats,
    pub vae_grid: GridMetrics,
    pub plain_grid: GridMetrics,
    pub vae_recon_mean: f64,
    pub plain_recon_mean: f64,
}

pub fn contrast_line(c: &Contrast) -> String {
    format!(
        "CONTRASTE: puntos z ~ N(0, I) que decodifican a un emoji reconocible: VAE {:.1} % contra autoencoder sin esquema variacional {:.1} % | a una cara plausible: VAE {:.1} % contra {:.1} % | a un hibrido plausible: VAE {:.1} % contra {:.1} % | salto medio entre celdas vecinas: VAE {:.1} px contra {:.1} px | reconstruccion media: VAE {:.2} px contra {:.2} px por emoji",
        100.0 * c.vae_prior.recognizable_frac,
        100.0 * c.plain_prior.recognizable_frac,
        100.0 * c.vae_prior.plausible_frac,
        100.0 * c.plain_prior.plausible_frac,
        100.0 * c.vae_prior.hybrid_frac,
        100.0 * c.plain_prior.hybrid_frac,
        c.vae_grid.mean_adjacent_step,
        c.plain_grid.mean_adjacent_step,
        c.vae_recon_mean,
        c.plain_recon_mean
    )
}

/// Muestras de la posterior q(z|x) por emoji de entrenamiento en el control negativo.
pub const CONTROL_POSTERIOR_SAMPLES: usize = 30;
/// Semilla del flujo de ruido del control negativo.
pub const CONTROL_SEED: u64 = 77;
/// Maximo de reconstrucciones (media posterior) de emojis de entrenamiento aceptadas como nuevas: ninguna.
pub const MAX_CONTROL_MEAN: f64 = 0.0;
/// Maximo de muestras de la posterior aceptadas como nuevas (calibrado: 4.4 a 5.3 % en las semillas 0 a 2).
pub const MAX_CONTROL_POSTERIOR: f64 = 0.10;

/// El predicado que esta correccion reemplaza, conservado solo como referencia del control: la distancia
/// global al emoji mas cercano es al menos `sep.min` y a lo sumo `2 * sep.near_radius`.
pub fn pixel_band_accepts(ds: &EmojiDataset, sep: &Separation, pattern: ArrayView1<f64>) -> bool {
    let distance = nearest_emoji(ds, pattern).1;
    distance >= sep.min && distance <= 2 * sep.near_radius
}

/// Resultado del control negativo: las reconstrucciones de los emojis de entrenamiento no deben aceptarse
/// como nuevas.
#[derive(Debug, Clone, PartialEq)]
pub struct ControlStats {
    pub n_mean: usize,
    /// Reconstrucciones de la media posterior aceptadas como nuevas (hibridas) por el juez por region.
    pub mean_accepted: usize,
    /// Las mismas reconstrucciones aceptadas por la banda de pixeles que esta correccion reemplaza.
    pub mean_accepted_by_band: usize,
    pub n_posterior: usize,
    /// Muestras de la posterior (reconstrucciones estocasticas) aceptadas como nuevas.
    pub posterior_accepted: usize,
}

/// Control negativo: (1) las reconstrucciones de la media posterior de los emojis de entrenamiento; (2)
/// `CONTROL_POSTERIOR_SAMPLES` reconstrucciones estocasticas por emoji, con ruido de `CONTROL_SEED`.
pub fn negative_control(vae: &Vae, style_judge: &StyleJudge) -> ControlStats {
    let (ds, sep) = (style_judge.dataset(), style_judge.separation());
    let n = ds.x.nrows();
    let recon = vae.reconstruct_mean(&ds.x);
    let mut mean_accepted = 0usize;
    let mut mean_accepted_by_band = 0usize;
    for row in recon.rows() {
        mean_accepted += usize::from(style_judge.judge(row).hybrid);
        mean_accepted_by_band += usize::from(pixel_band_accepts(ds, sep, row));
    }
    let (mu, logvar) = vae.encode_params(&ds.x);
    let mut rng = ChaCha8Rng::seed_from_u64(CONTROL_SEED);
    let mut posterior_accepted = 0usize;
    for _ in 0..CONTROL_POSTERIOR_SAMPLES {
        let eps = standard_normal_matrix(&mut rng, n, vae.latent_dim);
        let decoded = vae.decode(&reparameterize(&mu, &logvar, &eps));
        posterior_accepted += decoded.rows().into_iter().filter(|r| style_judge.judge(*r).hybrid).count();
    }
    ControlStats {
        n_mean: n,
        mean_accepted,
        mean_accepted_by_band,
        n_posterior: n * CONTROL_POSTERIOR_SAMPLES,
        posterior_accepted,
    }
}

pub fn control_line(c: &ControlStats) -> String {
    format!(
        "CONTROL NEGATIVO: reconstrucciones de los {} emojis del entrenamiento aceptadas como nuevas: {}/{} (criterio anterior de banda de pixeles: {}/{}) | muestras de la posterior q(z|x), {CONTROL_POSTERIOR_SAMPLES} por emoji: {:.1} % aceptadas como nuevas",
        c.n_mean,
        c.mean_accepted,
        c.n_mean,
        c.mean_accepted_by_band,
        c.n_mean,
        100.0 * c.posterior_accepted as f64 / c.n_posterior.max(1) as f64
    )
}

#[derive(Debug, Clone, PartialEq)]
pub struct AcceptanceCheck {
    pub name: &'static str,
    pub value: f64,
    pub threshold: f64,
    /// `true`: el valor debe ser al menos el umbral; `false`: a lo sumo.
    pub at_least: bool,
    pub pass: bool,
}

fn check(name: &'static str, value: f64, threshold: f64, at_least: bool) -> AcceptanceCheck {
    let pass = if at_least { value >= threshold } else { value <= threshold };
    AcceptanceCheck { name, value, threshold, at_least, pass }
}

/// Las once verificaciones de aceptacion del resultado generativo, en orden fijo.
pub fn acceptance_checks(
    recon: &ReconSummary,
    history: &VaeHistory,
    grid: &GridMetrics,
    prior: &PriorStats,
    plain_prior: &PriorStats,
    control: &ControlStats,
    sep: &Separation,
) -> Vec<AcceptanceCheck> {
    vec![
        check("reconstruccion media (px por emoji)", recon.mean_wrong, MAX_MEAN_WRONG_PX, false),
        check("KL final (nats)", history.kl.last().copied().unwrap_or(0.0), MIN_KL, true),
        check("celdas reconocibles de la grilla", grid.recognizable_frac, MIN_GRID_RECOGNIZABLE, true),
        check(
            "salto medio entre celdas vecinas (px)",
            grid.mean_adjacent_step,
            MAX_ADJACENT_FACTOR * sep.near_radius as f64,
            false,
        ),
        check("sorteos del prior reconocibles", prior.recognizable_frac, MIN_PRIOR_RECOGNIZABLE, true),
        check("sorteos del prior hibridos y plausibles", prior.hybrid_frac, MIN_NEW_RATE, true),
        check(
            "ventaja sobre el autoencoder simple (sorteos reconocibles)",
            prior.recognizable_frac - plain_prior.recognizable_frac,
            MIN_CONTRAST_GAP,
            true,
        ),
        check(
            "ventaja sobre el autoencoder simple (sorteos plausibles)",
            prior.plausible_frac - plain_prior.plausible_frac,
            MIN_VALID_GAP,
            true,
        ),
        check("combinaciones distintas entre los hibridos del prior", prior.hybrid_keys as f64, GALLERY_SIZE as f64, true),
        check(
            "control negativo: reconstrucciones del entrenamiento aceptadas como nuevas",
            control.mean_accepted as f64 / control.n_mean.max(1) as f64,
            MAX_CONTROL_MEAN,
            false,
        ),
        check(
            "control negativo: muestras de la posterior aceptadas como nuevas",
            control.posterior_accepted as f64 / control.n_posterior.max(1) as f64,
            MAX_CONTROL_POSTERIOR,
            false,
        ),
    ]
}

pub fn render_acceptance(checks: &[AcceptanceCheck]) -> String {
    checks
        .iter()
        .map(|c| {
            format!(
                "ACEPTACION {}: {} = {:.3} (umbral {} {:.3})",
                if c.pass { "OK" } else { "FALLA" },
                c.name,
                c.value,
                if c.at_least { ">=" } else { "<=" },
                c.threshold
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}
