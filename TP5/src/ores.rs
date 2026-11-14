//! Dataset del Ej 2 (VAE-01): las 18 texturas reales de los bloques de mena de Minecraft 1.21.4, de
//! 16 x 16 pixeles RGB. Son 8 minerales en piedra y en pizarra profunda mas el oro y el cuarzo del
//! Nether, asi que cada bloque tiene dos atributos: su mineral y su roca. Las texturas son de Mojang y
//! no se versionan: `scripts/descargar_menas.sh` las baja del cliente oficial, verifica sus hashes y las
//! deja en `datos/menas`, junto con las 3 rocas base (piedra, pizarra profunda y netherrack), que el
//! juez del Ej 2 usa para reconocer la roca. Modulo numerico: no dibuja ni parsea argumentos.

use std::collections::HashSet;
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use ndarray::{Array2, ArrayView1};

pub const ORE_SIDE: usize = 16;
pub const ORE_CHANNELS: usize = 3;
pub const ORE_PIXELS: usize = ORE_SIDE * ORE_SIDE;
/// Valores por bloque: los 3 canales de cada pixel, en orden (fila, columna, canal).
pub const ORE_VALUES: usize = ORE_PIXELS * ORE_CHANNELS;
pub const ORE_COUNT: usize = 18;

/// Minerales, en el orden de los indices de `ORE_BLOCKS`.
pub const ORE_MINERALS: [&str; 9] =
    ["carbon", "hierro", "cobre", "oro", "redstone", "esmeralda", "lapis", "diamante", "cuarzo"];
/// Rocas, en el orden de los indices de `ORE_BLOCKS` y de `ROCK_TEXTURES`.
pub const ORE_ROCKS: [&str; 3] = ["piedra", "pizarra", "nether"];
/// Textura base de cada roca (archivo sin `.png`).
pub const ROCK_TEXTURES: [&str; 3] = ["stone", "deepslate", "netherrack"];

/// Los 18 bloques del dataset: archivo de la textura (sin `.png`), mineral y roca.
pub const ORE_BLOCKS: [(&str, usize, usize); ORE_COUNT] = [
    ("coal_ore", 0, 0),
    ("iron_ore", 1, 0),
    ("copper_ore", 2, 0),
    ("gold_ore", 3, 0),
    ("redstone_ore", 4, 0),
    ("emerald_ore", 5, 0),
    ("lapis_ore", 6, 0),
    ("diamond_ore", 7, 0),
    ("deepslate_coal_ore", 0, 1),
    ("deepslate_iron_ore", 1, 1),
    ("deepslate_copper_ore", 2, 1),
    ("deepslate_gold_ore", 3, 1),
    ("deepslate_redstone_ore", 4, 1),
    ("deepslate_emerald_ore", 5, 1),
    ("deepslate_lapis_ore", 6, 1),
    ("deepslate_diamond_ore", 7, 1),
    ("nether_gold_ore", 3, 2),
    ("nether_quartz_ore", 8, 2),
];

/// Dos colores coinciden si ningun canal difiere en mas de 32 niveles (de 255).
pub const COLOR_TOLERANCE: f64 = 32.0 / 255.0;

/// Comando que deja las texturas en su lugar; los errores de carga lo nombran.
pub const DOWNLOAD_COMMAND: &str = "bash scripts/descargar_menas.sh";

#[derive(Debug, Clone)]
pub struct OreDataset {
    /// Nombres `{mineral}-{roca}`, en el orden de `ORE_BLOCKS`.
    pub names: Vec<String>,
    pub mineral: Vec<usize>,
    pub rock: Vec<usize>,
    /// 18 x 768 valores en [0, 1].
    pub x: Array2<f64>,
    /// 3 x 768: las texturas base de las rocas, en el orden de `ORE_ROCKS`.
    pub bases: Array2<f64>,
}

impl OreDataset {
    /// Combinaciones (roca, mineral) presentes en el dataset.
    pub fn combinations(&self) -> HashSet<(usize, usize)> {
        self.rock.iter().copied().zip(self.mineral.iter().copied()).collect()
    }
}

/// Directorio de las texturas: `TP5_MENAS` si esta definida, si no `datos/menas` dentro de TP5.
pub fn ore_dir() -> PathBuf {
    match std::env::var_os("TP5_MENAS") {
        Some(dir) => PathBuf::from(dir),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("datos").join("menas"),
    }
}

/// Lee un PNG de 16 x 16 (gris, gris con alfa, paleta, RGB o RGBA, de 1 a 16 bits) como 768 valores RGB
/// en [0, 1]. El alfa se descarta: las texturas de mena son opacas.
pub fn read_texture(path: &Path) -> Result<Vec<f64>> {
    let file = File::open(path)
        .with_context(|| format!("no se pudo abrir {}; corra `{DOWNLOAD_COMMAND}`", path.display()))?;
    let mut decoder = png::Decoder::new(file);
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().with_context(|| format!("{} no es un PNG valido", path.display()))?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).with_context(|| format!("no se pudo decodificar {}", path.display()))?;
    if (info.width as usize, info.height as usize) != (ORE_SIDE, ORE_SIDE) {
        bail!("{} mide {}x{} y se esperaba {ORE_SIDE}x{ORE_SIDE}", path.display(), info.width, info.height);
    }
    let channels = match info.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Indexed => bail!("{}: la paleta no se expandio", path.display()),
    };
    let mut out = Vec::with_capacity(ORE_VALUES);
    for p in 0..ORE_PIXELS {
        let px = &buf[p * channels..(p + 1) * channels];
        let rgb = if channels < 3 { [px[0]; 3] } else { [px[0], px[1], px[2]] };
        out.extend(rgb.iter().map(|&v| v as f64 / 255.0));
    }
    Ok(out)
}

/// Carga los 18 bloques y las 3 rocas base desde `dir`.
pub fn load_ores_from(dir: &Path) -> Result<OreDataset> {
    let mut data = Vec::with_capacity(ORE_COUNT * ORE_VALUES);
    let (mut names, mut mineral, mut rock) = (Vec::new(), Vec::new(), Vec::new());
    for (file, m, r) in ORE_BLOCKS {
        data.extend(read_texture(&dir.join(format!("{file}.png")))?);
        names.push(format!("{}-{}", ORE_MINERALS[m], ORE_ROCKS[r]));
        mineral.push(m);
        rock.push(r);
    }
    let mut bases = Vec::with_capacity(ORE_ROCKS.len() * ORE_VALUES);
    for file in ROCK_TEXTURES {
        bases.extend(read_texture(&dir.join(format!("{file}.png")))?);
    }
    Ok(OreDataset {
        names,
        mineral,
        rock,
        x: Array2::from_shape_vec((ORE_COUNT, ORE_VALUES), data).map_err(|e| anyhow!("forma del dataset: {e}"))?,
        bases: Array2::from_shape_vec((ORE_ROCKS.len(), ORE_VALUES), bases)
            .map_err(|e| anyhow!("forma de las rocas: {e}"))?,
    })
}

/// Carga el dataset desde `ore_dir()`.
pub fn load_ores() -> Result<OreDataset> {
    load_ores_from(&ore_dir())
}

/// Error medio por canal entre dos imagenes, en niveles de 0 a 255.
pub fn color_distance(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    255.0 * a.iter().zip(b.iter()).map(|(p, q)| (p - q).abs()).sum::<f64>() / a.len() as f64
}

/// El pixel `p` coincide en las dos imagenes: ningun canal difiere en mas de `COLOR_TOLERANCE`.
pub fn pixel_matches(a: ArrayView1<f64>, b: ArrayView1<f64>, p: usize) -> bool {
    (0..ORE_CHANNELS).all(|c| (a[p * ORE_CHANNELS + c] - b[p * ORE_CHANNELS + c]).abs() <= COLOR_TOLERANCE)
}

/// Pixeles (de 256) que no coinciden entre las dos imagenes.
pub fn wrong_pixels(a: ArrayView1<f64>, b: ArrayView1<f64>) -> usize {
    (0..ORE_PIXELS).filter(|&p| !pixel_matches(a, b, p)).count()
}

/// Pixeles incorrectos de cada reconstruccion respecto de su bloque.
pub fn wrong_pixel_counts(target: &Array2<f64>, recon: &Array2<f64>) -> Vec<usize> {
    target.rows().into_iter().zip(recon.rows()).map(|(t, r)| wrong_pixels(t, r)).collect()
}

/// Indice del bloque del dataset a menor `color_distance` (el primero ante empates) y esa distancia.
pub fn nearest_ore(ds: &OreDataset, image: ArrayView1<f64>) -> (usize, f64) {
    let mut best = (0usize, f64::INFINITY);
    for (i, row) in ds.x.rows().into_iter().enumerate() {
        let d = color_distance(row, image);
        if d < best.1 {
            best = (i, d);
        }
    }
    best
}

/// Separacion del dataset: distancia (en niveles) de cada bloque a su vecino mas cercano.
#[derive(Debug, Clone, PartialEq)]
pub struct Separation {
    pub min: f64,
    pub median: f64,
}

pub fn separation(ds: &OreDataset) -> Separation {
    let n = ds.x.nrows();
    let mut nearest: Vec<f64> = (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| j != i)
                .map(|j| color_distance(ds.x.row(i), ds.x.row(j)))
                .fold(f64::INFINITY, f64::min)
        })
        .collect();
    nearest.sort_by(|a, b| a.total_cmp(b));
    let median = if n % 2 == 1 { nearest[n / 2] } else { (nearest[n / 2 - 1] + nearest[n / 2]) / 2.0 };
    Separation { min: nearest[0], median }
}

/// Pixeles de `image` que coinciden con la textura base de la roca `rock`.
pub fn background_match(ds: &OreDataset, image: ArrayView1<f64>, rock: usize) -> usize {
    let base = ds.bases.row(rock);
    (0..ORE_PIXELS).filter(|&p| pixel_matches(image, base, p)).count()
}

/// Nombre de una combinacion: minerales unidos por `+`, un guion y la roca (`redstone+diamante-piedra`).
pub fn combination_name(minerals: &[usize], rock: usize) -> String {
    let m: Vec<&str> = minerals.iter().map(|&k| ORE_MINERALS[k]).collect();
    format!("{}-{}", m.join("+"), ORE_ROCKS[rock])
}
