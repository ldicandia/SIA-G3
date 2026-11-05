//! Regiones del dataset (ojos / boca / marco), pisos de ruido y veredicto por region del `StyleJudge`.
//! Pruebas puras sobre imagenes sinteticas: no entrenan ningun modelo.

use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use tp5::emoji::{
    build_emojis, eye_style_of, mouth_style_of, region_hamming, style_regions, EmojiDataset, StyleRegions, EMOJI_EYES,
    EMOJI_MOUTHS,
};
use tp5::vae_eval::{noise_floors, regions_line, NoiseFloors, RegionVerdict, StyleJudge, StyleVerdict, BLEND_SLACK};

const ZERO: NoiseFloors = NoiseFloors { eyes: 0, mouth: 0, frame: 0 };

fn on(v: f64) -> bool {
    v >= 0.5
}

fn flip_value(v: f64) -> f64 {
    if on(v) {
        0.0
    } else {
        1.0
    }
}

fn flip(img: &mut Array1<f64>, pixels: &[usize]) {
    for &k in pixels {
        img[k] = flip_value(img[k]);
    }
}

fn eyes_row(e: usize) -> usize {
    e * 6
}

/// Distancia de la region a cada uno de los 6 estilos, por bucles simples.
fn brute_dists(ds: &EmojiDataset, img: &Array1<f64>, pixels: &[usize], row_of: &dyn Fn(usize) -> usize) -> Vec<usize> {
    (0..6)
        .map(|s| {
            let mut d = 0;
            for &k in pixels {
                if on(img[k]) != on(ds.x[[row_of(s), k]]) {
                    d += 1;
                }
            }
            d
        })
        .collect()
}

fn brute_gap(ds: &EmojiDataset, a: usize, b: usize, pixels: &[usize], row_of: &dyn Fn(usize) -> usize) -> usize {
    let mut d = 0;
    for &k in pixels {
        if on(ds.x[[row_of(a), k]]) != on(ds.x[[row_of(b), k]]) {
            d += 1;
        }
    }
    d
}

fn brute_region(
    ds: &EmojiDataset,
    img: &Array1<f64>,
    pixels: &[usize],
    floor: usize,
    row_of: &dyn Fn(usize) -> usize,
) -> RegionVerdict {
    let d = brute_dists(ds, img, pixels, row_of);
    let mut order: Vec<usize> = (0..6).collect();
    order.sort_by_key(|&s| (d[s], s));
    let (style, second_style) = (order[0], order[1]);
    let off_style = d[style] > floor;
    let gap = brute_gap(ds, style, second_style, pixels, row_of);
    let between = (d[style] + d[second_style]) as f64 <= BLEND_SLACK * gap as f64;
    RegionVerdict {
        style,
        distance: d[style],
        second_style,
        second_distance: d[second_style],
        off_style,
        blend_ok: !off_style || between,
    }
}

fn brute_verdict(
    ds: &EmojiDataset,
    regions: &StyleRegions,
    floors: NoiseFloors,
    img: &Array1<f64>,
) -> StyleVerdict {
    let eyes = brute_region(ds, img, &regions.eyes, floors.eyes, &|s| s * 6);
    let mouth = brute_region(ds, img, &regions.mouth, floors.mouth, &|s| s);
    let mut frame_distance = 0;
    for &k in &regions.frame {
        if on(img[k]) != on(ds.x[[0, k]]) {
            frame_distance += 1;
        }
    }
    let plausible = frame_distance <= floors.frame && eyes.blend_ok && mouth.blend_ok;
    let hybrid = (eyes.off_style || mouth.off_style) && plausible;
    let mut margin = 0;
    if eyes.off_style {
        margin = margin.max(eyes.distance - floors.eyes);
    }
    if mouth.off_style {
        margin = margin.max(mouth.distance - floors.mouth);
    }
    StyleVerdict { eyes, mouth, frame_distance, plausible, hybrid, margin }
}

#[test]
fn style_regions_partition_the_pixels_and_isolate_each_factor() {
    let ds = build_emojis();
    let r = style_regions(&ds);
    assert_eq!((r.eyes.len(), r.mouth.len(), r.frame.len()), (46, 62, 468));
    assert_eq!(r.eyes.len() + r.mouth.len() + r.frame.len(), 576);
    for list in [&r.eyes, &r.mouth, &r.frame] {
        assert!(list.windows(2).all(|w| w[0] < w[1]), "lista no ascendente");
    }
    let mut seen = vec![0usize; 576];
    for list in [&r.eyes, &r.mouth, &r.frame] {
        for &k in list.iter() {
            seen[k] += 1;
        }
    }
    assert!(seen.iter().all(|&c| c == 1), "las regiones deben ser disjuntas y cubrir los 576 pixeles");
    for i in 0..36 {
        for j in 0..36 {
            if eye_style_of(i) == eye_style_of(j) {
                assert!(r.eyes.iter().all(|&k| on(ds.x[[i, k]]) == on(ds.x[[j, k]])));
                assert_eq!(region_hamming(ds.x.row(i), ds.x.row(j), &r.eyes), 0);
            }
            if mouth_style_of(i) == mouth_style_of(j) {
                assert!(r.mouth.iter().all(|&k| on(ds.x[[i, k]]) == on(ds.x[[j, k]])));
            }
            assert!(r.frame.iter().all(|&k| on(ds.x[[i, k]]) == on(ds.x[[j, k]])));
        }
    }
}

#[test]
fn eye_and_mouth_style_indices_follow_the_dataset_order() {
    let ds = build_emojis();
    for i in 0..36 {
        let name = format!("{}-{}", EMOJI_EYES[eye_style_of(i)], EMOJI_MOUTHS[mouth_style_of(i)]);
        assert_eq!(ds.names[i], name);
    }
}

#[test]
fn region_hamming_counts_only_the_given_pixels() {
    let a = Array1::from(vec![0.0, 1.0, 0.4, 0.6, 0.9, 0.1]);
    let b = Array1::from(vec![1.0, 1.0, 0.6, 0.9, 0.2, 0.3]);
    // difieren (tras binarizar en 0.5) en las posiciones 0, 2, 4
    assert_eq!(region_hamming(a.view(), b.view(), &[0, 1, 2, 3, 4, 5]), 3);
    assert_eq!(region_hamming(a.view(), b.view(), &[1, 3, 5]), 0);
    assert_eq!(region_hamming(a.view(), b.view(), &[0, 4]), 2);
    assert_eq!(region_hamming(a.view(), b.view(), &[]), 0);
}

#[test]
fn noise_floors_are_the_per_region_maximum_over_the_training_reconstructions() {
    let ds = build_emojis();
    let r = style_regions(&ds);
    let mut recon: Array2<f64> = ds.x.clone();
    assert_eq!(noise_floors(&ds, &r, &recon), ZERO);
    for &k in r.eyes.iter().take(2) {
        recon[[3, k]] = flip_value(recon[[3, k]]);
    }
    for &k in r.mouth.iter().take(5) {
        recon[[10, k]] = flip_value(recon[[10, k]]);
    }
    for &k in r.frame.iter().take(1) {
        recon[[20, k]] = flip_value(recon[[20, k]]);
    }
    assert_eq!(noise_floors(&ds, &r, &recon), NoiseFloors { eyes: 2, mouth: 5, frame: 1 });
}

#[test]
fn dataset_rows_are_on_style_and_never_hybrid_even_with_zero_floors() {
    let ds = build_emojis();
    let judge = StyleJudge::with_floors(&ds, ZERO);
    for i in 0..36 {
        let v = judge.judge(ds.x.row(i));
        assert_eq!(v.eyes.style, eye_style_of(i));
        assert_eq!(v.mouth.style, mouth_style_of(i));
        assert_eq!((v.eyes.distance, v.mouth.distance), (0, 0));
        assert!(!v.eyes.off_style && !v.mouth.off_style);
        assert!(v.plausible && !v.hybrid);
        assert_eq!(v.margin, 0);
    }
}

#[test]
fn every_combination_of_known_styles_is_already_a_dataset_emoji() {
    let ds = build_emojis();
    let r = style_regions(&ds);
    let judge = StyleJudge::with_floors(&ds, ZERO);
    for a in 0..36 {
        // la region de un prototipo depende solo de su propio estilo
        let proto = eyes_row(eye_style_of(a));
        assert_eq!(region_hamming(ds.x.row(a), ds.x.row(proto), &r.eyes), 0);
        assert_eq!(region_hamming(ds.x.row(a), ds.x.row(mouth_style_of(a)), &r.mouth), 0);
        for b in 0..36 {
            let mut img = ds.x.row(0).to_owned();
            for &k in &r.eyes {
                img[k] = ds.x[[a, k]];
            }
            for &k in &r.mouth {
                img[k] = ds.x[[b, k]];
            }
            let row = 6 * eye_style_of(a) + mouth_style_of(b);
            assert_eq!(region_hamming(img.view(), ds.x.row(row), &r.eyes), 0);
            assert_eq!(region_hamming(img.view(), ds.x.row(row), &r.mouth), 0);
            assert_eq!(tp5::emoji::hamming(img.view(), ds.x.row(row)), 0, "la combinacion {a} x {b} ya es un emoji");
            let v = judge.judge(img.view());
            assert_eq!((v.eyes.style, v.eyes.distance), (eye_style_of(a), 0));
            assert_eq!((v.mouth.style, v.mouth.distance), (mouth_style_of(b), 0));
            assert!(!v.eyes.off_style && !v.mouth.off_style && !v.hybrid);
        }
    }
}

#[test]
fn a_blend_of_two_eye_styles_is_a_hybrid() {
    let ds = build_emojis();
    let r = style_regions(&ds);
    let mut best = (usize::MAX, 0usize, 1usize);
    for a in 0..6 {
        for b in (a + 1)..6 {
            let d = brute_gap(&ds, a, b, &r.eyes, &|s| s * 6);
            if d < best.0 {
                best = (d, a, b);
            }
        }
    }
    let (_, a, b) = best;
    let differing: Vec<usize> =
        r.eyes.iter().copied().filter(|&k| on(ds.x[[eyes_row(a), k]]) != on(ds.x[[eyes_row(b), k]])).collect();
    let mut img = ds.x.row(0).to_owned();
    for &k in &r.eyes {
        img[k] = ds.x[[eyes_row(a), k]];
    }
    for &k in differing.iter().take(differing.len() / 2) {
        img[k] = ds.x[[eyes_row(b), k]];
    }
    let judge = StyleJudge::with_floors(&ds, ZERO);
    let v = judge.judge(img.view());
    let expected = brute_dists(&ds, &img, &r.eyes, &|s| s * 6).into_iter().min().unwrap();
    assert!(expected > 0);
    assert!(v.eyes.off_style && v.eyes.blend_ok, "{v:?}");
    assert_eq!(v.eyes.distance, expected);
    assert!(!v.mouth.off_style && v.plausible && v.hybrid);
    assert_eq!(v.margin, expected);
}

#[test]
fn the_floor_decides_off_style() {
    let ds = build_emojis();
    let r = style_regions(&ds);
    let judge = StyleJudge::with_floors(&ds, NoiseFloors { eyes: 1, mouth: 14, frame: 2 });
    for (k, off) in [(1usize, false), (2usize, true)] {
        let mut img = ds.x.row(0).to_owned();
        flip(&mut img, &r.eyes[..k]);
        let nearest = brute_dists(&ds, &img, &r.eyes, &|s| s * 6).into_iter().min().unwrap();
        let v = judge.judge(img.view());
        assert_eq!(nearest, k);
        assert_eq!(v.eyes.distance, nearest);
        assert_eq!(v.eyes.style, 0);
        assert_eq!(v.eyes.off_style, off, "{v:?}");
    }
}

#[test]
fn a_broken_frame_or_a_region_between_no_styles_is_not_plausible() {
    let ds = build_emojis();
    let r = style_regions(&ds);
    let floors = NoiseFloors { eyes: 1, mouth: 14, frame: 2 };
    let judge = StyleJudge::with_floors(&ds, floors);
    let protos = |s: usize| s * 6;

    // Regla de mezcla: el primer k para el que la region no es un estilo ni queda entre dos.
    let mut found = None;
    for k in 2..=r.eyes.len() {
        let mut img = ds.x.row(0).to_owned();
        flip(&mut img, &r.eyes[..k]);
        let d = brute_dists(&ds, &img, &r.eyes, &protos);
        let mut order: Vec<usize> = (0..6).collect();
        order.sort_by_key(|&s| (d[s], s));
        let (n1, n2) = (order[0], order[1]);
        let gap = brute_gap(&ds, n1, n2, &r.eyes, &protos);
        if d[n1] > floors.eyes && (d[n1] + d[n2]) as f64 > BLEND_SLACK * gap as f64 {
            found = Some((k, img, n1, n2, d[n1], d[n2]));
            break;
        }
    }
    let (_, img, n1, n2, d1, d2) = found.expect("ningun k produce una region fuera de estilo y fuera de mezcla");
    let v = judge.judge(img.view());
    assert!(v.eyes.off_style && !v.eyes.blend_ok, "{v:?}");
    assert_eq!((v.eyes.style, v.eyes.distance, v.eyes.second_style, v.eyes.second_distance), (n1, d1, n2, d2));
    assert!(!v.plausible && !v.hybrid);

    // Regla del marco: ojos fuera de estilo pero el marco roto.
    let mut img = ds.x.row(0).to_owned();
    flip(&mut img, &r.eyes[..4]);
    flip(&mut img, &r.frame[..3]);
    let nearest = brute_dists(&ds, &img, &r.eyes, &protos).into_iter().min().unwrap();
    assert!(nearest > floors.eyes, "garantizado por la desigualdad triangular, pero se verifica");
    let mut frame_distance = 0;
    for &k in &r.frame {
        if on(img[k]) != on(ds.x[[0, k]]) {
            frame_distance += 1;
        }
    }
    assert!(frame_distance > floors.frame);
    let v = judge.judge(img.view());
    assert_eq!(v.frame_distance, frame_distance);
    assert!(v.eyes.off_style);
    assert!(!v.plausible && !v.hybrid);
}

#[test]
fn verdicts_match_an_independent_brute_force_over_perturbed_dataset_rows() {
    let ds = build_emojis();
    let r = style_regions(&ds);
    let floors = NoiseFloors { eyes: 3, mouth: 14, frame: 2 };
    let judge = StyleJudge::with_floors(&ds, floors);
    let (mut hybrid, mut blend_fail, mut frame_fail, mut on_style) = (0usize, 0usize, 0usize, 0usize);
    for i in 0..36 {
        let mut rng = ChaCha8Rng::seed_from_u64(i as u64);
        let (mut pe, mut pm, mut pf) = (r.eyes.clone(), r.mouth.clone(), r.frame.clone());
        pe.shuffle(&mut rng);
        pm.shuffle(&mut rng);
        pf.shuffle(&mut rng);
        for &fe in &[0usize, 2, 4, 6, 10, 20] {
            for &fm in &[0usize, 5, 10, 16, 25] {
                for &ff in &[0usize, 2, 3, 6] {
                    let mut img = ds.x.row(i).to_owned();
                    flip(&mut img, &pe[..fe]);
                    flip(&mut img, &pm[..fm]);
                    flip(&mut img, &pf[..ff]);
                    let got = judge.judge(img.view());
                    let want = brute_verdict(&ds, &r, floors, &img);
                    assert_eq!(got, want, "fila {i}, flips {fe}/{fm}/{ff}");
                    hybrid += usize::from(got.hybrid);
                    blend_fail += usize::from(
                        got.frame_distance <= floors.frame
                            && ((got.eyes.off_style && !got.eyes.blend_ok) || (got.mouth.off_style && !got.mouth.blend_ok)),
                    );
                    frame_fail += usize::from(got.frame_distance > floors.frame);
                    on_style += usize::from(!got.eyes.off_style && !got.mouth.off_style && got.plausible);
                }
            }
        }
    }
    assert!(hybrid > 0, "el barrido no contiene ningun hibrido: ampliar los flips");
    assert!(blend_fail > 0, "el barrido no contiene ninguna region fuera de mezcla: ampliar los flips");
    assert!(frame_fail > 0, "el barrido no contiene ningun marco roto: ampliar los flips");
    assert!(on_style > 0, "el barrido no contiene ninguna imagen en estilo: ampliar los flips");
}

#[test]
fn regions_line_reports_sizes_floors_and_separation() {
    let ds = build_emojis();
    let floors = NoiseFloors { eyes: 3, mouth: 14, frame: 2 };
    let judge = StyleJudge::with_floors(&ds, floors);
    let (se, sm) = judge.min_style_separation();
    let line = regions_line(&judge);
    let head = format!(
        "REGIONES: ojos 46 px, boca 62 px, marco 468 px | piso de ruido de reconstruccion (maximo sobre los 36 emojis): ojos {} px, boca {} px, marco {} px",
        floors.eyes, floors.mouth, floors.frame
    );
    assert!(line.starts_with(&head), "{line}");
    assert!(line.ends_with(&format!("separacion minima entre estilos: ojos {se} px, boca {sm} px")), "{line}");
    assert_eq!(judge.regions().eyes.len(), 46);
    assert_eq!(judge.floors(), floors);
}
