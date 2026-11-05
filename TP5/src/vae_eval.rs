//! Evaluacion generativa del VAE (Ej 2, VAE-02 y VAE-03): juicio de las imagenes decodificadas contra
//! el dataset con umbrales derivados del propio dataset, busqueda de un emoji nuevo en el prior y
//! estadisticas de los sorteos del prior. Modulo numerico: no dibuja ni parsea argumentos.

use ndarray::{Array1, Array2, ArrayView1};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::autoencoder::{train, Autoencoder, TrainConfig};
use crate::emoji::{
    hamming, nearest_emoji, region_hamming, separation, style_regions, EmojiDataset, Separation, StyleRegions,
    EMOJI_MOUTHS, EMOJI_PIXELS,
};
use crate::report::TOLERANCE_PX;
use crate::vae::{standard_normal_matrix, ReconSummary, Vae, VaeConfig, VaeHistory};

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
pub const MIN_NEW_RATE: f64 = 0.10;
pub const MIN_CONTRAST_GAP: f64 = 0.3;

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
        "CONTRASTE: puntos z ~ N(0, I) que decodifican a un emoji reconocible: VAE {:.1} % contra autoencoder sin esquema variacional {:.1} % | salto medio entre celdas vecinas: VAE {:.1} px contra {:.1} px | reconstruccion media: VAE {:.2} px contra {:.2} px por emoji",
        100.0 * c.vae_prior.recognizable_frac,
        100.0 * c.plain_prior.recognizable_frac,
        c.vae_grid.mean_adjacent_step,
        c.plain_grid.mean_adjacent_step,
        c.vae_recon_mean,
        c.plain_recon_mean
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

/// Las siete verificaciones de aceptacion del resultado generativo, en orden fijo.
pub fn acceptance_checks(
    recon: &ReconSummary,
    history: &VaeHistory,
    grid: &GridMetrics,
    prior: &PriorStats,
    plain_prior: &PriorStats,
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
        check("sorteos del prior nuevos y plausibles", prior.new_plausible_frac, MIN_NEW_RATE, true),
        check(
            "ventaja sobre el autoencoder simple (sorteos reconocibles)",
            prior.recognizable_frac - plain_prior.recognizable_frac,
            MIN_CONTRAST_GAP,
            true,
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
