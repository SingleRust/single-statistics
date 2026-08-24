#![cfg(feature = "enrichment")]
use single_statistics::enrichment::gsea;

/// metric = 100..0, so gene 0 is top-ranked.
fn ranked(n: usize) -> Vec<f64> {
    (0..n).map(|i| (n - i) as f64).collect()
}

#[test]
fn top_ranked_set_scores_positive() {
    let m = ranked(200);
    let top: Vec<usize> = (0..20).collect();
    let r = gsea(&m, &[top], 200, 1.0, 42).unwrap();

    assert_eq!(r[0].size, 20);
    assert!(r[0].es > 0.9, "es = {}", r[0].es);
    assert!(r[0].p_value < 0.05, "p = {}", r[0].p_value);
    assert!(r[0].nes > 0.0);
}

#[test]
fn bottom_ranked_set_scores_negative() {
    let m = ranked(200);
    let bottom: Vec<usize> = (180..200).collect();
    let r = gsea(&m, &[bottom], 200, 1.0, 42).unwrap();

    assert!(r[0].es < -0.9, "es = {}", r[0].es);
    assert!(r[0].p_value < 0.05, "p = {}", r[0].p_value);
}

#[test]
fn scattered_set_is_not_enriched() {
    let m = ranked(200);
    let spread: Vec<usize> = (0..20).map(|i| i * 10).collect();
    let r = gsea(&m, &[spread], 200, 1.0, 42).unwrap();

    assert!(r[0].es.abs() < 0.5, "es = {}", r[0].es);
    assert!(r[0].p_value > 0.05, "p = {}", r[0].p_value);
}

#[test]
fn leading_edge_holds_the_drivers() {
    let m = ranked(200);
    let top: Vec<usize> = (0..20).collect();
    let r = gsea(&m, &[top], 100, 1.0, 42).unwrap();

    // Whole set sits at the head, so all of it drives the peak.
    assert_eq!(r[0].leading_edge.len(), 20);
    assert!(r[0].leading_edge.iter().all(|g| *g < 20));
}

#[test]
fn same_seed_reproduces() {
    let m = ranked(200);
    let s: Vec<usize> = (0..20).collect();
    let a = gsea(&m, std::slice::from_ref(&s), 100, 1.0, 7).unwrap();
    let b = gsea(&m, &[s], 100, 1.0, 7).unwrap();
    assert_eq!(a[0].p_value, b[0].p_value);
    assert_eq!(a[0].nes, b[0].nes);
}

#[test]
fn weight_zero_is_unweighted() {
    let m = ranked(200);
    let top: Vec<usize> = (0..20).collect();
    let r = gsea(&m, &[top], 100, 0.0, 42).unwrap();
    // Classic KS still finds a head-loaded set.
    assert!(r[0].es > 0.8, "es = {}", r[0].es);
}

#[test]
fn adjusts_across_pathways() {
    let m = ranked(500);
    let sets: Vec<Vec<usize>> = (0..10).map(|i| (i * 20..i * 20 + 20).collect()).collect();
    let r = gsea(&m, &sets, 100, 1.0, 42).unwrap();

    assert_eq!(r.len(), 10);
    for x in &r {
        assert!(x.adjusted_p_value >= x.p_value - 1e-12);
        assert!((0.0..=1.0).contains(&x.adjusted_p_value));
    }
    assert!(r[0].es > r[5].es, "head set should beat a middle one");
}

#[test]
fn dedups_and_drops_out_of_range_genes() {
    let m = ranked(100);
    let messy = vec![0, 0, 1, 1, 2, 500, 900];
    let r = gsea(&m, &[messy], 50, 1.0, 42).unwrap();
    assert_eq!(r[0].size, 3, "expect genes 0,1,2 only");
}

#[test]
fn handles_degenerate_input() {
    assert!(gsea(&[], &[vec![0]], 10, 1.0, 1).is_err());
    assert!(gsea(&[1.0, 2.0], &[], 10, 1.0, 1).is_err());

    // Empty set, and a set covering everything: no score either way.
    let m = ranked(50);
    let r = gsea(&m, &[vec![], (0..50).collect()], 10, 1.0, 1).unwrap();
    assert_eq!(r[0].es, 0.0);
    assert_eq!(r[1].es, 0.0);
    assert_eq!(r[0].p_value, 1.0);
}

#[test]
fn zero_permutations_skips_the_null() {
    let m = ranked(100);
    let r = gsea(&m, &[(0..10).collect()], 0, 1.0, 1).unwrap();
    assert!(r[0].es > 0.0, "es still computed");
    assert!(r[0].nes.is_nan());
    assert_eq!(r[0].p_value, 1.0);
}

/// The O(k) score must match a brute-force O(n) walk exactly, across many
/// random sets and rankings.
#[test]
fn fast_es_matches_naive_walk() {
    fn naive(order: &[usize], w: &[f64], in_set: &[bool], k: usize) -> f64 {
        let n = order.len();
        if k == 0 || k == n { return 0.0; }
        let n_r: f64 = order.iter().zip(w).filter(|(g, _)| in_set[**g]).map(|(_, x)| *x).sum();
        if n_r <= 0.0 { return 0.0; }
        let miss = 1.0 / (n - k) as f64;
        let (mut run, mut best) = (0.0f64, 0.0f64);
        for (&g, &x) in order.iter().zip(w) {
            run += if in_set[g] { x / n_r } else { -miss };
            if run.abs() > best.abs() { best = run; }
        }
        best
    }

    let mut seed = 12345u64;
    let mut rng = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };

    for trial in 0..300 {
        let n = 20 + (rng() % 200) as usize;
        let k = 1 + (rng() % (n as u64 - 1)) as usize;
        let metric: Vec<f64> = (0..n).map(|_| (rng() % 2000) as f64 / 100.0 - 10.0).collect();

        let mut in_set = vec![false; n];
        let mut chosen = 0;
        while chosen < k {
            let g = (rng() % n as u64) as usize;
            if !in_set[g] { in_set[g] = true; chosen += 1; }
        }
        let set: Vec<usize> = (0..n).filter(|&g| in_set[g]).collect();

        // Reproduce the ordering gsea uses internally.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by(|&a, &b| metric[b].partial_cmp(&metric[a]).unwrap());
        let w: Vec<f64> = order.iter().map(|&g| metric[g].abs()).collect();

        let expected = naive(&order, &w, &in_set, k);
        let got = gsea(&metric, &[set], 0, 1.0, 1).unwrap()[0].es;

        assert!(
            (got - expected).abs() < 1e-12,
            "trial {}: n={} k={} fast={} naive={}", trial, n, k, got, expected
        );
    }
}
