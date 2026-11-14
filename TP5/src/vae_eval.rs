//! Evaluacion generativa del VAE (Ej 2, VAE-02 y VAE-03): juicio de las imagenes decodificadas contra
//! el dataset de menas, busqueda de un bloque nuevo en el prior y estadisticas de los sorteos del prior.
//! Modulo numerico: no dibuja ni parsea argumentos.
//!
//! El juez describe una imagen por sus dos atributos. La roca es la textura base (piedra, pizarra
//! profunda o netherrack) con la que coinciden mas pixeles. Los minerales salen de descomponer la imagen
//! como mezcla convexa de los 18 bloques del dataset (pesos no negativos que suman 1, por minimos
//! cuadrados): un mineral esta presente si los bloques de ese mineral suman al menos `MIN_SHARE` del peso.
//! Una imagen es plausible si su fondo coincide con una roca y la mezcla la explica tan bien como explica
//! las reconstrucciones del propio modelo; es un bloque nuevo si ademas su combinacion de roca y minerales
//! no existe en el dataset (por ejemplo, redstone y diamante en la misma piedra).

use std::collections::{HashMap, HashSet};

use ndarray::{Array1, Array2, ArrayView1};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::autoencoder::{train, Autoencoder, TrainConfig};
use crate::ores::{
    background_match, color_distance, combination_name, nearest_ore, wrong_pixel_counts, OreDataset, Separation,
    ORE_CHANNELS, ORE_MINERALS, ORE_PIXELS, ORE_ROCKS, ORE_VALUES,
};
use crate::report::TOLERANCE_PX;
use crate::vae::{reparameterize, standard_normal_matrix, ReconSummary, Vae, VaeConfig, VaeHistory};

/// Un salto entre celdas vecinas es grande si supera `LARGE_STEP_FACTOR` veces la separacion mediana del dataset.
pub const LARGE_STEP_FACTOR: f64 = 3.0;
/// Sorteos por defecto del prior.
pub const DEFAULT_DRAWS: usize = 1000;
/// Tope de `--draws`.
pub const MAX_DRAWS: usize = 100_000;
/// Cantidad maxima de muestras aceptadas que se guardan para la galeria.
pub const GALLERY_SIZE: usize = 8;
/// Filas por tanda al decodificar sorteos: la memoria no crece con `--draws`.
const DRAW_CHUNK: usize = 1000;

/// Un mineral esta presente si sus bloques suman al menos este peso de la mezcla.
pub const MIN_SHARE: f64 = 0.25;
/// El fondo es de una roca si coinciden con su textura base mas de la mitad de los pixeles.
pub const MIN_BACKGROUND: usize = ORE_PIXELS / 2 + 1;
/// El residuo de la mezcla debe ser a lo sumo esta fraccion de la separacion minima del dataset: la mezcla
/// explica la imagen mejor que la mitad de la distancia entre los dos bloques mas parecidos.
pub const RESIDUAL_FRACTION: f64 = 0.5;
/// Iteraciones del descenso por gradiente proyectado de la mezcla.
const MIXTURE_ITERS: usize = 2000;

/// Valida el argumento `--draws`: entre 1 y `MAX_DRAWS`.
pub fn validate_draws(n: usize) -> anyhow::Result<usize> {
    if n == 0 || n > MAX_DRAWS {
        anyhow::bail!("el numero de sorteos debe estar entre 1 y {MAX_DRAWS} (se recibio {n})");
    }
    Ok(n)
}

/// Proyeccion euclidea sobre el simplex {w >= 0, suma 1} (algoritmo por ordenamiento).
pub fn project_simplex(v: &Array1<f64>) -> Array1<f64> {
    let mut u: Vec<f64> = v.to_vec();
    u.sort_by(|a, b| b.total_cmp(a));
    let mut cumulative = 0.0;
    let mut theta = 0.0;
    for (k, &value) in u.iter().enumerate() {
        cumulative += value;
        let t = (cumulative - 1.0) / (k + 1) as f64;
        if value - t > 0.0 {
            theta = t;
        }
    }
    v.mapv(|x| (x - theta).max(0.0))
}

/// Descomposicion de imagenes como mezcla convexa de los bloques del dataset.
#[derive(Debug, Clone)]
pub struct Mixer {
    x: Array2<f64>,
    gram: Array2<f64>,
    step: f64,
}

impl Mixer {
    pub fn new(ds: &OreDataset) -> Mixer {
        let x = ds.x.clone();
        let gram = x.dot(&x.t());
        // Paso 1/L con L el mayor autovalor de la matriz de Gram (iteracion de potencia determinista).
        let mut v = Array1::from_elem(gram.nrows(), 1.0);
        let mut lambda = 1.0;
        for _ in 0..200 {
            let w = gram.dot(&v);
            lambda = w.dot(&w).sqrt();
            v = w / lambda;
        }
        Mixer { x, gram, step: 1.0 / lambda }
    }

    /// Pesos (uno por bloque del dataset, no negativos y de suma 1) que minimizan el error cuadratico de
    /// la mezcla, por descenso por gradiente proyectado desde la mezcla uniforme.
    pub fn weights(&self, image: ArrayView1<f64>) -> Array1<f64> {
        let n = self.x.nrows();
        let b = self.x.dot(&image);
        let mut w = Array1::from_elem(n, 1.0 / n as f64);
        for _ in 0..MIXTURE_ITERS {
            let grad = self.gram.dot(&w) - &b;
            w = project_simplex(&(&w - &(grad * self.step)));
        }
        w
    }

    /// Error medio por canal (niveles de 0 a 255) entre la imagen y la mezcla.
    pub fn residual(&self, image: ArrayView1<f64>, weights: &Array1<f64>) -> f64 {
        let mix = self.x.t().dot(weights);
        color_distance(mix.view(), image)
    }
}

/// Atributos de una imagen.
#[derive(Debug, Clone, PartialEq)]
pub struct Verdict {
    /// Roca cuya textura base coincide con mas pixeles (ante empates, la de menor indice).
    pub rock: usize,
    /// Pixeles (de 256) que coinciden con esa roca.
    pub background: usize,
    /// Peso de cada mineral en la mezcla.
    pub shares: Vec<f64>,
    /// Minerales con al menos `MIN_SHARE` del peso, en orden de indice.
    pub minerals: Vec<usize>,
    /// Error de la mezcla, en niveles.
    pub residual: f64,
    /// Contraste de la imagen, en niveles (ver `contrast`).
    pub contrast: f64,
    /// Fondo de una roca, residuo dentro del tope, contraste al menos el de la reconstruccion menos
    /// contrastada del modelo y algun mineral presente.
    pub plausible: bool,
    /// Plausible y con una combinacion (roca, mineral) del dataset.
    pub known: bool,
    /// Plausible y con una combinacion ausente del dataset: el predicado que acepta un bloque nuevo.
    pub new_block: bool,
}

impl Verdict {
    pub fn combination(&self) -> String {
        combination_name(&self.minerals, self.rock)
    }

    /// Equilibrio de la mezcla: el menor peso entre los minerales presentes (0 si no hay ninguno). Un
    /// bloque nuevo con equilibrio alto muestra claramente sus dos (o mas) minerales.
    pub fn balance(&self) -> f64 {
        self.minerals.iter().map(|&m| self.shares[m]).reduce(f64::min).unwrap_or(0.0)
    }
}

/// Pisos del juez, medidos sobre las reconstrucciones del modelo de los 18 bloques de entrenamiento.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JudgeFloors {
    /// Menor coincidencia de fondo con su roca.
    pub background: usize,
    /// Mayor residuo de la mezcla.
    pub residual: f64,
    /// Mayor peso de un mineral ajeno (su control: debe quedar por debajo de `MIN_SHARE`).
    pub foreign_share: f64,
    /// Menor contraste: es el contraste minimo que se exige a una imagen plausible.
    pub contrast: f64,
}

#[derive(Debug, Clone)]
pub struct OreJudge {
    ds: OreDataset,
    sep: Separation,
    mixer: Mixer,
    floors: JudgeFloors,
    known: HashSet<(usize, Vec<usize>)>,
}

impl OreJudge {
    /// `recon` tiene en la fila `i` la reconstruccion del bloque de entrenamiento `i`.
    pub fn new(ds: &OreDataset, recon: &Array2<f64>) -> OreJudge {
        let mixer = Mixer::new(ds);
        let mut floors =
            JudgeFloors { background: usize::MAX, residual: 0.0, foreign_share: 0.0, contrast: f64::INFINITY };
        for (i, row) in recon.rows().into_iter().enumerate() {
            let (_, background) = best_rock(ds, row);
            let w = mixer.weights(row);
            let shares = mineral_shares(ds, &w);
            floors.background = floors.background.min(background);
            floors.residual = floors.residual.max(mixer.residual(row, &w));
            floors.contrast = floors.contrast.min(contrast(row));
            for (m, &s) in shares.iter().enumerate() {
                if m != ds.mineral[i] {
                    floors.foreign_share = floors.foreign_share.max(s);
                }
            }
        }
        let known = ds.rock.iter().zip(&ds.mineral).map(|(&r, &m)| (r, vec![m])).collect();
        OreJudge { ds: ds.clone(), sep: crate::ores::separation(ds), mixer: Mixer::new(ds), floors, known }
    }

    pub fn floors(&self) -> JudgeFloors {
        self.floors
    }

    pub fn dataset(&self) -> &OreDataset {
        &self.ds
    }

    pub fn separation(&self) -> &Separation {
        &self.sep
    }

    /// Contraste minimo de una imagen plausible: el de la reconstruccion menos contrastada.
    pub fn min_contrast(&self) -> f64 {
        self.floors.contrast
    }

    pub fn residual_limit(&self) -> f64 {
        RESIDUAL_FRACTION * self.sep.min
    }

    pub fn judge(&self, image: ArrayView1<f64>) -> Verdict {
        let (rock, background) = best_rock(&self.ds, image);
        let w = self.mixer.weights(image);
        let shares = mineral_shares(&self.ds, &w);
        let minerals: Vec<usize> = (0..ORE_MINERALS.len()).filter(|&m| shares[m] >= MIN_SHARE).collect();
        let residual = self.mixer.residual(image, &w);
        let contrast = contrast(image);
        let plausible = background >= MIN_BACKGROUND
            && residual <= self.residual_limit()
            && contrast >= self.floors.contrast
            && !minerals.is_empty();
        let in_dataset = self.known.contains(&(rock, minerals.clone()));
        Verdict {
            rock,
            background,
            contrast,
            shares,
            minerals,
            residual,
            plausible,
            known: plausible && in_dataset,
            new_block: plausible && !in_dataset,
        }
    }
}

/// Contraste de una imagen: desvio absoluto medio de cada canal respecto del color medio de la imagen, en
/// niveles de 0 a 255. Una imagen lisa (por ejemplo, una mezcla borrosa) tiene contraste bajo.
pub fn contrast(image: ArrayView1<f64>) -> f64 {
    let n = ORE_PIXELS as f64;
    let mut mean = [0.0; ORE_CHANNELS];
    for p in 0..ORE_PIXELS {
        for (c, m) in mean.iter_mut().enumerate() {
            *m += image[p * ORE_CHANNELS + c] / n;
        }
    }
    let total: f64 = (0..ORE_PIXELS)
        .map(|p| (0..ORE_CHANNELS).map(|c| (image[p * ORE_CHANNELS + c] - mean[c]).abs()).sum::<f64>())
        .sum();
    255.0 * total / (n * ORE_CHANNELS as f64)
}

fn best_rock(ds: &OreDataset, image: ArrayView1<f64>) -> (usize, usize) {
    let mut best = (0usize, 0usize);
    for r in 0..ORE_ROCKS.len() {
        let m = background_match(ds, image, r);
        if m > best.1 || r == 0 {
            best = (r, m);
        }
    }
    best
}

fn mineral_shares(ds: &OreDataset, w: &Array1<f64>) -> Vec<f64> {
    let mut shares = vec![0.0; ORE_MINERALS.len()];
    for (i, &m) in ds.mineral.iter().enumerate() {
        shares[m] += w[i];
    }
    shares
}

/// Linea `JUEZ:` con los umbrales y lo que miden sobre las reconstrucciones del modelo.
pub fn judge_line(judge: &OreJudge) -> String {
    let f = judge.floors();
    format!(
        "JUEZ: roca = textura base con mas pixeles coincidentes, al menos {MIN_BACKGROUND} de 256 (reconstrucciones: al menos {}) | minerales = los de al menos {:.0} % del peso de la mezcla convexa de los 18 bloques (mayor peso ajeno en las reconstrucciones: {:.1} %) | residuo de la mezcla a lo sumo {:.2} niveles, la mitad de la separacion minima (reconstrucciones: a lo sumo {:.2}) | contraste al menos {:.1} niveles, el de la reconstruccion menos contrastada",
        f.background,
        100.0 * MIN_SHARE,
        100.0 * f.foreign_share,
        judge.residual_limit(),
        f.residual,
        judge.min_contrast()
    )
}

#[derive(Debug, Clone)]
pub struct NewOre {
    /// Posicion (desde 0) del sorteo en el flujo semillado.
    pub draw_index: usize,
    pub z: Array1<f64>,
    /// Salida del decodificador (RGB en [0, 1]).
    pub image: Array1<f64>,
    /// Bloque del dataset mas cercano y su distancia en niveles.
    pub nearest: (usize, f64),
    pub verdict: Verdict,
}

#[derive(Debug, Clone)]
pub struct SampleSearch {
    pub seed: u64,
    pub max_draws: usize,
    pub examined: usize,
    pub floors: JudgeFloors,
    pub residual_limit: f64,
    /// Sorteos aceptados como bloques nuevos.
    pub candidates: usize,
    /// Combinaciones nuevas distintas entre ellos.
    pub distinct_keys: usize,
    /// A lo sumo `GALLERY_SIZE`, una por combinacion, en el orden de `gallery_order`.
    pub accepted: Vec<NewOre>,
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

/// Orden de la galeria: mayor equilibrio, luego menor residuo, luego sorteo anterior.
fn gallery_order(a: &NewOre, b: &NewOre) -> std::cmp::Ordering {
    b.verdict
        .balance()
        .total_cmp(&a.verdict.balance())
        .then(a.verdict.residual.total_cmp(&b.verdict.residual))
        .then(a.draw_index.cmp(&b.draw_index))
}

/// Recorre los `max_draws` sorteos del flujo semillado y juzga cada imagen decodificada. De los bloques
/// nuevos conserva, por cada combinacion, el primero segun `gallery_order` (mezcla mas equilibrada,
/// luego menor residuo, luego sorteo anterior). La galeria son las `GALLERY_SIZE` primeras combinaciones
/// en ese mismo orden; el bloque principal es `accepted[0]`.
pub fn search_new_ore(vae: &Vae, ds: &OreDataset, seed: u64, max_draws: usize) -> SampleSearch {
    let judge = OreJudge::new(ds, &vae.reconstruct_mean(&ds.x));
    let mut best: HashMap<String, NewOre> = HashMap::new();
    let mut candidates = 0usize;
    let decode = |z: &Array2<f64>| vae.decode(z);
    let examined = walk_prior(&decode, vae.latent_dim, seed, max_draws, &mut |idx, z, img| {
        let verdict = judge.judge(img);
        if verdict.new_block {
            candidates += 1;
            let key = verdict.combination();
            let nearest = nearest_ore(ds, img);
            let sample = NewOre { draw_index: idx, z: z.to_owned(), image: img.to_owned(), nearest, verdict };
            let better = best.get(&key).is_none_or(|current| gallery_order(&sample, current).is_lt());
            if better {
                best.insert(key, sample);
            }
        }
        true
    });
    let distinct_keys = best.len();
    let mut accepted: Vec<NewOre> = best.into_values().collect();
    accepted.sort_by(gallery_order);
    accepted.truncate(GALLERY_SIZE);
    SampleSearch {
        seed,
        max_draws,
        examined,
        floors: judge.floors(),
        residual_limit: judge.residual_limit(),
        candidates,
        distinct_keys,
        accepted,
    }
}

const NEW_CRITERION: &str = "fondo de una roca conocida, mezcla de bloques del dataset con residuo dentro del tope, contraste de un bloque real y una combinacion de roca y minerales que no existe en el dataset";

/// Pesos de los minerales presentes, como `redstone 53 %, diamante 47 %`.
pub fn shares_text(v: &Verdict) -> String {
    v.minerals
        .iter()
        .map(|&m| format!("{} {:.0} %", ORE_MINERALS[m], 100.0 * v.shares[m]))
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn new_ore_line(search: &SampleSearch, ds: &OreDataset) -> String {
    match search.accepted.first() {
        Some(h) => format!(
            "BLOQUE NUEVO: z=({:.4}, {:.4}) sorteo {} de {} | {} (combinacion ausente del dataset) | mezcla: {} | fondo: {} de 256 px de {} (minimo {MIN_BACKGROUND}) | residuo {:.2} niveles (tope {:.2}) | bloque mas cercano: {} a {:.1} niveles | criterio: {NEW_CRITERION}",
            h.z[0],
            h.z[1],
            h.draw_index + 1,
            search.examined,
            h.verdict.combination(),
            shares_text(&h.verdict),
            h.verdict.background,
            ORE_ROCKS[h.verdict.rock],
            h.verdict.residual,
            search.residual_limit,
            ds.names[h.nearest.0],
            h.nearest.1
        ),
        None => format!("BLOQUE NUEVO: ninguno en {} sorteos (criterio: {NEW_CRITERION})", search.max_draws),
    }
}

/// Galeria de las muestras aceptadas: una linea de encabezado y una por muestra, con dos espacios al inicio.
pub fn render_gallery(search: &SampleSearch, ds: &OreDataset) -> String {
    let mut lines = vec![format!(
        "GALERIA: {} sorteos aceptados como bloques nuevos de {}, {} combinaciones nuevas distintas; se muestran {} ordenadas por equilibrio de la mezcla",
        search.candidates,
        search.examined,
        search.distinct_keys,
        search.accepted.len()
    )];
    for (i, s) in search.accepted.iter().enumerate() {
        lines.push(format!(
            "  nuevo {}: sorteo {} | {} | mezcla: {} | fondo {} px | residuo {:.2} niveles | bloque mas cercano {} a {:.1} niveles",
            i + 1,
            s.draw_index + 1,
            s.verdict.combination(),
            shares_text(&s.verdict),
            s.verdict.background,
            s.verdict.residual,
            ds.names[s.nearest.0],
            s.nearest.1
        ));
    }
    lines.join("\n")
}

/// Participacion de los sorteos del prior en cada categoria.
#[derive(Debug, Clone, PartialEq)]
pub struct PriorStats {
    pub n: usize,
    /// Plausibles con una combinacion del dataset.
    pub recognizable_frac: f64,
    pub plausible_frac: f64,
    /// Plausibles con una combinacion ausente del dataset.
    pub new_frac: f64,
    /// Combinaciones nuevas distintas.
    pub new_keys: usize,
}

/// Estadisticas de los primeros `n` sorteos del flujo semillado, decodificados por `decode` (asi el
/// autoencoder simple reutiliza la misma funcion con su propio juez).
pub fn prior_stats(
    decode: &dyn Fn(&Array2<f64>) -> Array2<f64>,
    latent_dim: usize,
    judge: &OreJudge,
    seed: u64,
    n: usize,
) -> PriorStats {
    let (mut known, mut plausible, mut new) = (0usize, 0usize, 0usize);
    let mut keys: HashSet<String> = HashSet::new();
    let examined = walk_prior(decode, latent_dim, seed, n, &mut |_, _, img| {
        let v = judge.judge(img);
        known += usize::from(v.known);
        plausible += usize::from(v.plausible);
        if v.new_block {
            new += 1;
            keys.insert(v.combination());
        }
        true
    });
    let total = examined.max(1) as f64;
    PriorStats {
        n: examined,
        recognizable_frac: known as f64 / total,
        plausible_frac: plausible as f64 / total,
        new_frac: new as f64 / total,
        new_keys: keys.len(),
    }
}

pub fn prior_line(s: &PriorStats) -> String {
    format!(
        "PRIOR: de {} puntos z ~ N(0, I): {:.1} % decodifican a un bloque plausible (fondo de una roca, mezcla de bloques del dataset y contraste de un bloque real), {:.1} % a un bloque reconocible (plausible y con una combinacion del dataset), {:.1} % a un bloque nuevo (plausible y con una combinacion ausente) en {} combinaciones distintas",
        s.n,
        100.0 * s.plausible_frac,
        100.0 * s.recognizable_frac,
        100.0 * s.new_frac,
        s.new_keys
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
    /// Celdas plausibles.
    pub plausible_frac: f64,
    /// Celdas reconocibles (plausibles con una combinacion del dataset).
    pub recognizable_frac: f64,
    /// Media de la distancia de color (niveles) entre celdas vecinas (horizontales y verticales).
    pub mean_adjacent_step: f64,
    pub max_adjacent_step: f64,
    /// Participacion de pares vecinos con un salto de mas de `LARGE_STEP_FACTOR` veces la separacion mediana.
    pub large_step_frac: f64,
}

pub fn manifold_metrics(decoded: &Array2<f64>, side: usize, judge: &OreJudge) -> GridMetrics {
    assert_eq!(decoded.nrows(), side * side, "la grilla decodificada debe tener side * side filas");
    let n_cells = side * side;
    let verdicts: Vec<Verdict> = decoded.rows().into_iter().map(|r| judge.judge(r)).collect();
    let mut steps = Vec::with_capacity(2 * side * (side - 1));
    for r in 0..side {
        for c in 0..side {
            let here = decoded.row(r * side + c);
            if c + 1 < side {
                steps.push(color_distance(here, decoded.row(r * side + c + 1)));
            }
            if r + 1 < side {
                steps.push(color_distance(here, decoded.row((r + 1) * side + c)));
            }
        }
    }
    let limit = LARGE_STEP_FACTOR * judge.separation().median;
    let large = steps.iter().filter(|&&s| s > limit).count();
    GridMetrics {
        side,
        n_cells,
        plausible_frac: verdicts.iter().filter(|v| v.plausible).count() as f64 / n_cells as f64,
        recognizable_frac: verdicts.iter().filter(|v| v.known).count() as f64 / n_cells as f64,
        mean_adjacent_step: steps.iter().sum::<f64>() / steps.len() as f64,
        max_adjacent_step: steps.iter().copied().fold(0.0, f64::max),
        large_step_frac: large as f64 / steps.len() as f64,
    }
}

pub fn manifold_line(m: &GridMetrics, sep: &Separation) -> String {
    format!(
        "MANIFOLD: {} celdas ({}x{}) sobre z en [-{:.1}, {:.1}]^2 | plausibles {:.1} %, reconocibles {:.1} % | salto medio entre celdas vecinas {:.1} niveles (maximo {:.1}; separacion mediana del dataset {:.1}) | saltos de mas de {:.1} niveles: {:.1} %",
        m.n_cells,
        m.side,
        m.side,
        GRID_HALF_RANGE,
        GRID_HALF_RANGE,
        100.0 * m.plausible_frac,
        100.0 * m.recognizable_frac,
        m.mean_adjacent_step,
        m.max_adjacent_step,
        sep.median,
        LARGE_STEP_FACTOR * sep.median,
        100.0 * m.large_step_frac
    )
}

// Umbrales de aceptacion: convenciones de este trabajo, documentadas y no ajustadas para aprobar.
/// Pixeles incorrectos por bloque (de 256) en la reconstruccion.
pub const MAX_MEAN_WRONG_PX: f64 = 12.0;
pub const MIN_KL: f64 = 1.0;
pub const MIN_GRID_PLAUSIBLE: f64 = 0.5;
pub const MAX_ADJACENT_FACTOR: f64 = 1.5;
pub const MIN_PRIOR_PLAUSIBLE: f64 = 0.6;
pub const MIN_NEW_RATE: f64 = 0.05;
/// Ventaja minima de sorteos plausibles del VAE sobre el autoencoder simple.
pub const MIN_VALID_GAP: f64 = 0.3;

/// Autoencoder simple con la misma arquitectura, optimizador, perdida, epocas y semilla que el VAE,
/// pero sin capa estocastica ni termino KL.
pub fn train_plain_ae(ds: &OreDataset, cfg: &VaeConfig) -> Autoencoder {
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    let mut ae = Autoencoder::new(&cfg.spec, ORE_VALUES, &mut rng);
    let tcfg = TrainConfig { epochs: cfg.epochs, eval_every: cfg.epochs, tolerance_px: TOLERANCE_PX, early_stop: false };
    train(&mut ae, &ds.x, &ds.x, cfg.recon, &cfg.optimizer, &tcfg);
    ae
}

/// Puntos de una grilla `side x side` sobre la caja de los codigos de entrenamiento del autoencoder
/// simple, con 10 % del rango de cada eje de margen a cada lado, en el orden y la orientacion de
/// `latent_grid` (z1 crece hacia la derecha, z2 decrece hacia abajo).
pub fn plain_grid_points(ae: &Autoencoder, ds: &OreDataset, side: usize) -> Array2<f64> {
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

/// Pixeles incorrectos medios por bloque de una reconstruccion.
pub fn mean_wrong_pixels(ds: &OreDataset, recon: &Array2<f64>) -> f64 {
    let counts = wrong_pixel_counts(&ds.x, recon);
    counts.iter().sum::<usize>() as f64 / counts.len() as f64
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
        "CONTRASTE: puntos z ~ N(0, I) que decodifican a un bloque plausible: VAE {:.1} % contra autoencoder sin esquema variacional {:.1} % | reconocible: VAE {:.1} % contra {:.1} % | nuevo: VAE {:.1} % contra {:.1} % | salto medio entre celdas vecinas: VAE {:.1} contra {:.1} niveles | reconstruccion media: VAE {:.2} px contra {:.2} px por bloque",
        100.0 * c.vae_prior.plausible_frac,
        100.0 * c.plain_prior.plausible_frac,
        100.0 * c.vae_prior.recognizable_frac,
        100.0 * c.plain_prior.recognizable_frac,
        100.0 * c.vae_prior.new_frac,
        100.0 * c.plain_prior.new_frac,
        c.vae_grid.mean_adjacent_step,
        c.plain_grid.mean_adjacent_step,
        c.vae_recon_mean,
        c.plain_recon_mean
    )
}

/// Muestras de la posterior q(z|x) por bloque de entrenamiento en el control negativo.
pub const CONTROL_POSTERIOR_SAMPLES: usize = 30;
/// Semilla del flujo de ruido del control negativo.
pub const CONTROL_SEED: u64 = 77;
/// Maximo de reconstrucciones (media posterior) de bloques de entrenamiento aceptadas como nuevas: ninguna.
pub const MAX_CONTROL_MEAN: f64 = 0.0;
/// Maximo de muestras de la posterior aceptadas como nuevas.
pub const MAX_CONTROL_POSTERIOR: f64 = 0.10;

/// Resultado del control negativo: las reconstrucciones de los bloques de entrenamiento no deben aceptarse
/// como nuevas.
#[derive(Debug, Clone, PartialEq)]
pub struct ControlStats {
    pub n_mean: usize,
    /// Reconstrucciones de la media posterior aceptadas como nuevas.
    pub mean_accepted: usize,
    /// Reconstrucciones de la media posterior juzgadas reconocibles con su propia combinacion.
    pub mean_own_combination: usize,
    pub n_posterior: usize,
    /// Muestras de la posterior (reconstrucciones estocasticas) aceptadas como nuevas.
    pub posterior_accepted: usize,
}

/// Control negativo: (1) las reconstrucciones de la media posterior de los bloques de entrenamiento; (2)
/// `CONTROL_POSTERIOR_SAMPLES` reconstrucciones estocasticas por bloque, con ruido de `CONTROL_SEED`.
pub fn negative_control(vae: &Vae, judge: &OreJudge) -> ControlStats {
    let ds = judge.dataset();
    let n = ds.x.nrows();
    let recon = vae.reconstruct_mean(&ds.x);
    let (mut mean_accepted, mut mean_own_combination) = (0usize, 0usize);
    for (i, row) in recon.rows().into_iter().enumerate() {
        let v = judge.judge(row);
        mean_accepted += usize::from(v.new_block);
        mean_own_combination += usize::from(v.known && v.rock == ds.rock[i] && v.minerals == vec![ds.mineral[i]]);
    }
    let (mu, logvar) = vae.encode_params(&ds.x);
    let mut rng = ChaCha8Rng::seed_from_u64(CONTROL_SEED);
    let mut posterior_accepted = 0usize;
    for _ in 0..CONTROL_POSTERIOR_SAMPLES {
        let eps = standard_normal_matrix(&mut rng, n, vae.latent_dim);
        let decoded = vae.decode(&reparameterize(&mu, &logvar, &eps));
        posterior_accepted += decoded.rows().into_iter().filter(|r| judge.judge(*r).new_block).count();
    }
    ControlStats {
        n_mean: n,
        mean_accepted,
        mean_own_combination,
        n_posterior: n * CONTROL_POSTERIOR_SAMPLES,
        posterior_accepted,
    }
}

pub fn control_line(c: &ControlStats) -> String {
    format!(
        "CONTROL NEGATIVO: reconstrucciones de los {} bloques del entrenamiento aceptadas como nuevas: {}/{} (reconocidas con su propia combinacion: {}/{}) | muestras de la posterior q(z|x), {CONTROL_POSTERIOR_SAMPLES} por bloque: {:.1} % aceptadas como nuevas",
        c.n_mean,
        c.mean_accepted,
        c.n_mean,
        c.mean_own_combination,
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

/// Las verificaciones de aceptacion del resultado generativo, en orden fijo.
pub fn acceptance_checks(
    recon: &ReconSummary,
    history: &VaeHistory,
    grid: &GridMetrics,
    prior: &PriorStats,
    plain_prior: &PriorStats,
    control: &ControlStats,
    judge: &OreJudge,
) -> Vec<AcceptanceCheck> {
    vec![
        check("reconstruccion media (px incorrectos por bloque)", recon.mean_wrong, MAX_MEAN_WRONG_PX, false),
        check("KL final (nats)", history.kl.last().copied().unwrap_or(0.0), MIN_KL, true),
        check("celdas plausibles de la grilla", grid.plausible_frac, MIN_GRID_PLAUSIBLE, true),
        check(
            "salto medio entre celdas vecinas (niveles)",
            grid.mean_adjacent_step,
            MAX_ADJACENT_FACTOR * judge.separation().median,
            false,
        ),
        check("sorteos del prior plausibles", prior.plausible_frac, MIN_PRIOR_PLAUSIBLE, true),
        check("sorteos del prior nuevos", prior.new_frac, MIN_NEW_RATE, true),
        check(
            "ventaja sobre el autoencoder simple (sorteos plausibles)",
            prior.plausible_frac - plain_prior.plausible_frac,
            MIN_VALID_GAP,
            true,
        ),
        check("combinaciones nuevas distintas en el prior", prior.new_keys as f64, GALLERY_SIZE as f64, true),
        check("peso de un mineral ajeno en las reconstrucciones", judge.floors().foreign_share, MIN_SHARE, false),
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
