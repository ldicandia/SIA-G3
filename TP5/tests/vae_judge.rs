//! Dataset de menas y juez de bloques del Ej 2 (`OreJudge`): carga de las texturas, distancias de color,
//! proyeccion al simplex, mezcla convexa y veredictos sobre los bloques originales y mezclas sinteticas.
//! No entrenan ningun modelo.

use std::collections::HashSet;

use ndarray::{array, Array1};

use tp5::ores::{
    background_match, color_distance, combination_name, load_ores, nearest_ore, separation, wrong_pixels, OreDataset,
    ORE_BLOCKS, ORE_COUNT, ORE_MINERALS, ORE_PIXELS, ORE_ROCKS, ORE_VALUES,
};
use tp5::vae_eval::{
    contrast, judge_line, project_simplex, Mixer, OreJudge, MIN_BACKGROUND, MIN_SHARE,
};

fn ores() -> OreDataset {
    load_ores().expect("dataset de menas: corra `bash scripts/descargar_menas.sh`")
}

fn index_of(ds: &OreDataset, name: &str) -> usize {
    ds.names.iter().position(|n| n == name).unwrap_or_else(|| panic!("{name} no esta en el dataset"))
}

fn blend(ds: &OreDataset, a: &str, b: &str, t: f64) -> Array1<f64> {
    &ds.x.row(index_of(ds, a)) * (1.0 - t) + &ds.x.row(index_of(ds, b)) * t
}

/// Juez con los pisos medidos sobre los propios originales (reconstruccion perfecta).
fn exact_judge(ds: &OreDataset) -> OreJudge {
    OreJudge::new(ds, &ds.x)
}

#[test]
fn dataset_has_18_named_blocks_with_their_attributes() {
    let ds = ores();
    assert_eq!(ds.x.dim(), (ORE_COUNT, ORE_VALUES));
    assert_eq!(ds.bases.dim(), (ORE_ROCKS.len(), ORE_VALUES));
    assert!(ds.x.iter().chain(ds.bases.iter()).all(|&v| (0.0..=1.0).contains(&v)));
    let unique: HashSet<&String> = ds.names.iter().collect();
    assert_eq!(unique.len(), ORE_COUNT);
    for (i, (_, m, r)) in ORE_BLOCKS.iter().enumerate() {
        assert_eq!((ds.mineral[i], ds.rock[i]), (*m, *r));
        assert_eq!(ds.names[i], format!("{}-{}", ORE_MINERALS[*m], ORE_ROCKS[*r]));
    }
    assert_eq!(ds.names[0], "carbon-piedra");
    assert_eq!(ds.names[17], "cuarzo-nether");
    assert_eq!(ds.combinations().len(), ORE_COUNT);
}

#[test]
fn loading_is_deterministic_and_fails_with_the_download_command() {
    let (a, b) = (ores(), ores());
    assert!(a.x.iter().zip(b.x.iter()).all(|(p, q)| p.to_bits() == q.to_bits()));
    let err = tp5::ores::load_ores_from(std::path::Path::new("no-existe")).unwrap_err().to_string();
    assert!(err.contains("descargar_menas.sh"), "{err}");
}

#[test]
fn every_block_differs_from_the_others_and_is_its_own_nearest() {
    let ds = ores();
    for i in 0..ORE_COUNT {
        assert_eq!(nearest_ore(&ds, ds.x.row(i)), (i, 0.0));
        for j in (i + 1)..ORE_COUNT {
            assert!(wrong_pixels(ds.x.row(i), ds.x.row(j)) > 0, "{} y {}", ds.names[i], ds.names[j]);
        }
    }
}

#[test]
fn separation_matches_an_independent_computation() {
    let ds = ores();
    let mut nearest = Vec::new();
    for i in 0..ORE_COUNT {
        let mut best = f64::INFINITY;
        for j in 0..ORE_COUNT {
            if i != j {
                let d: f64 = (0..ORE_VALUES).map(|k| (ds.x[[i, k]] - ds.x[[j, k]]).abs()).sum::<f64>() * 255.0
                    / ORE_VALUES as f64;
                best = best.min(d);
            }
        }
        nearest.push(best);
    }
    nearest.sort_by(|a, b| a.total_cmp(b));
    let s = separation(&ds);
    assert!((s.min - nearest[0]).abs() < 1e-9);
    assert!((s.median - (nearest[8] + nearest[9]) / 2.0).abs() < 1e-9);
}

#[test]
fn color_distance_is_the_mean_absolute_error_in_levels() {
    let a = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let b = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    assert!((color_distance(a.view(), b.view()) - 127.5).abs() < 1e-12);
}

#[test]
fn every_block_matches_its_own_rock_on_most_pixels() {
    let ds = ores();
    for i in 0..ORE_COUNT {
        let own = background_match(&ds, ds.x.row(i), ds.rock[i]);
        assert!(own >= MIN_BACKGROUND, "{}: {own} px", ds.names[i]);
        for (r, rock) in ORE_ROCKS.iter().enumerate() {
            if r != ds.rock[i] {
                assert!(background_match(&ds, ds.x.row(i), r) < own, "{} se parece mas a {rock}", ds.names[i]);
            }
        }
    }
}

#[test]
fn simplex_projection_is_nonnegative_sums_to_one_and_fixes_simplex_points() {
    let p = project_simplex(&array![0.5, 0.3, 0.2]);
    assert!(p.iter().zip([0.5, 0.3, 0.2]).all(|(a, b)| (a - b).abs() < 1e-12));
    let q = project_simplex(&array![2.0, -1.0, 0.0]);
    assert!((q[0] - 1.0).abs() < 1e-12 && q[1] == 0.0 && q[2] == 0.0);
    let r = project_simplex(&array![0.4, 0.4, -3.0, 0.9]);
    assert!(r.iter().all(|&v| v >= 0.0));
    assert!((r.sum() - 1.0).abs() < 1e-12);
}

#[test]
fn the_mixture_of_a_dataset_block_is_that_block() {
    let ds = ores();
    let mixer = Mixer::new(&ds);
    for i in 0..ORE_COUNT {
        let w = mixer.weights(ds.x.row(i));
        assert!(w[i] > 0.97, "{}: peso propio {}", ds.names[i], w[i]);
        assert!(mixer.residual(ds.x.row(i), &w) < 1.0);
    }
}

#[test]
fn every_original_is_recognized_with_its_own_combination() {
    let ds = ores();
    let judge = exact_judge(&ds);
    assert!(judge.floors().foreign_share < MIN_SHARE);
    for i in 0..ORE_COUNT {
        let v = judge.judge(ds.x.row(i));
        assert_eq!((v.rock, v.minerals.clone()), (ds.rock[i], vec![ds.mineral[i]]), "{}", ds.names[i]);
        assert!(v.plausible && v.known && !v.new_block, "{}: {v:?}", ds.names[i]);
        assert_eq!(v.combination(), ds.names[i]);
    }
}

#[test]
fn an_even_blend_of_two_minerals_on_the_same_rock_is_a_new_block() {
    let ds = ores();
    let judge = exact_judge(&ds);
    for (a, b) in [("redstone-piedra", "esmeralda-piedra"), ("lapis-pizarra", "diamante-pizarra")] {
        let v = judge.judge(blend(&ds, a, b, 0.5).view());
        let expected: Vec<usize> = {
            let mut m = vec![ds.mineral[index_of(&ds, a)], ds.mineral[index_of(&ds, b)]];
            m.sort();
            m
        };
        assert_eq!(v.rock, ds.rock[index_of(&ds, a)]);
        assert_eq!(v.minerals, expected, "{a} + {b}: {v:?}");
        assert!(v.plausible && v.new_block && !v.known, "{a} + {b}: {v:?}");
        assert_eq!(v.combination(), combination_name(&expected, v.rock));
    }
}

#[test]
fn a_slight_blend_keeps_the_dominant_block() {
    let ds = ores();
    let judge = exact_judge(&ds);
    let v = judge.judge(blend(&ds, "oro-piedra", "lapis-piedra", 0.1).view());
    assert_eq!(v.combination(), "oro-piedra");
    assert!(v.known && !v.new_block, "{v:?}");
}

#[test]
fn a_blend_across_rocks_or_a_flat_image_is_not_plausible() {
    let ds = ores();
    let judge = exact_judge(&ds);
    let across = judge.judge(blend(&ds, "diamante-piedra", "cuarzo-nether", 0.5).view());
    assert!(!across.plausible && !across.new_block, "{across:?}");
    // Un gris liso coincide con la piedra en la mayoria de los pixeles y queda cerca de la mezcla de
    // todos los bloques: lo descarta el contraste.
    let flat = Array1::from_elem(ORE_VALUES, 0.5);
    let v = judge.judge(flat.view());
    assert!(!v.plausible && !v.new_block, "{v:?}");
    assert_eq!(v.contrast, 0.0);
    assert!(v.contrast < judge.min_contrast());
}

#[test]
fn contrast_floor_is_the_least_contrasted_block() {
    let ds = ores();
    let judge = exact_judge(&ds);
    let least = ds.x.rows().into_iter().map(contrast).fold(f64::INFINITY, f64::min);
    assert_eq!(judge.min_contrast(), least);
    // contraste recalculado a mano para un bloque
    let row = ds.x.row(0);
    let mean: Vec<f64> = (0..3).map(|c| (0..256).map(|p| row[3 * p + c]).sum::<f64>() / 256.0).collect();
    let by_hand = 255.0 * (0..768).map(|k| (row[k] - mean[k % 3]).abs()).sum::<f64>() / 768.0;
    assert!((contrast(row) - by_hand).abs() < 1e-9);
}

#[test]
fn background_threshold_is_more_than_half_the_pixels() {
    assert_eq!(MIN_BACKGROUND, ORE_PIXELS / 2 + 1);
}

#[test]
fn judge_line_reports_the_thresholds() {
    let ds = ores();
    let judge = exact_judge(&ds);
    let line = judge_line(&judge);
    assert!(line.starts_with("JUEZ: roca = textura base con mas pixeles coincidentes, al menos 129 de 256"), "{line}");
    assert!(line.contains("minerales = los de al menos 25 % del peso"), "{line}");
    assert!(line.contains(&format!("residuo de la mezcla a lo sumo {:.2} niveles", judge.residual_limit())), "{line}");
}

#[test]
fn combinations_present_are_18_of_27() {
    let ds = ores();
    let present = ds.combinations();
    assert_eq!(ORE_MINERALS.len() * ORE_ROCKS.len(), 27);
    assert_eq!(present.len(), 18);
    let nether = ORE_ROCKS.iter().position(|&r| r == "nether").unwrap();
    let quartz = ORE_MINERALS.iter().position(|&m| m == "cuarzo").unwrap();
    // Faltan los 7 minerales que no estan en el Nether y el cuarzo en piedra y en pizarra.
    let missing_in_nether = (0..ORE_MINERALS.len()).filter(|&m| !present.contains(&(nether, m))).count();
    let quartz_elsewhere = (0..ORE_ROCKS.len()).filter(|&r| r != nether && present.contains(&(r, quartz))).count();
    assert_eq!(missing_in_nether, 7);
    assert_eq!(quartz_elsewhere, 0);
}
