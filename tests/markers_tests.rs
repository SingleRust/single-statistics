//! Marker statistics and the negative binomial test.
//! Reference values are hand-computed and shown in the comments.

use single_statistics::testing::inference::discrete::negative_binomial_test;
use single_statistics::testing::inference::nonparametric::mann_whitney_optimized;
use single_statistics::testing::markers::{filter_marker_candidates, marker_statistics};
use single_statistics::testing::utils::SparseMatrixRef;
use single_statistics::testing::Alternative;

// ------------------------------------------------------------------ marker statistics

#[test]
fn marker_statistics_are_hand_verifiable() {
    // 1 gene x 6 cells. Group 1 = cells 0,1,2 ; Group 2 = cells 3,4,5
    // Gene 0 values: [0, 2, 4, 0, 0, 6]
    //   group1: sum = 6, mean = 2.0, 2 of 3 expressed -> pct = 2/3
    //   group2: sum = 6, mean = 2.0, 1 of 3 expressed -> pct = 1/3
    let indptr = vec![0usize, 3];
    let indices = vec![1usize, 2, 5];
    let values = vec![2.0f64, 4.0, 6.0];

    let smr = SparseMatrixRef::new(&indptr, &indices, &values, 1, 6);
    let stats = marker_statistics(smr, &[0, 1, 2], &[3, 4, 5], 1.0).unwrap();

    assert_eq!(stats.len(), 1);
    let s = &stats[0];
    assert!((s.mean_group1 - 2.0).abs() < 1e-12, "mean1 = {}", s.mean_group1);
    assert!((s.mean_group2 - 2.0).abs() < 1e-12, "mean2 = {}", s.mean_group2);
    assert!((s.pct_group1 - 2.0 / 3.0).abs() < 1e-12, "pct1 = {}", s.pct_group1);
    assert!((s.pct_group2 - 1.0 / 3.0).abs() < 1e-12, "pct2 = {}", s.pct_group2);
    // Equal means -> log2((2+1)/(2+1)) = 0
    assert!(s.log2_fold_change.abs() < 1e-12, "log2fc = {}", s.log2_fold_change);
    assert!((s.delta_pct() - 1.0 / 3.0).abs() < 1e-12);
}

#[test]
fn auroc_is_one_for_perfect_separation() {
    // group1 strictly below group2 -> AUROC = 0 (group 1 never ranks higher)
    let indptr = vec![0usize, 6];
    let indices: Vec<usize> = (0..6).collect();
    let values = vec![1.0f64, 2.0, 3.0, 10.0, 11.0, 12.0];

    let smr = SparseMatrixRef::new(&indptr, &indices, &values, 1, 6);
    let stats = marker_statistics(smr, &[0, 1, 2], &[3, 4, 5], 1.0).unwrap();
    assert!((stats[0].auroc - 0.0).abs() < 1e-12, "auroc = {}", stats[0].auroc);

    // Reversed groups -> AUROC = 1
    let smr = SparseMatrixRef::new(&indptr, &indices, &values, 1, 6);
    let stats = marker_statistics(smr, &[3, 4, 5], &[0, 1, 2], 1.0).unwrap();
    assert!((stats[0].auroc - 1.0).abs() < 1e-12, "auroc = {}", stats[0].auroc);
}

#[test]
fn auroc_is_half_for_identical_groups() {
    let indptr = vec![0usize, 6];
    let indices: Vec<usize> = (0..6).collect();
    let values = vec![5.0f64; 6];

    let smr = SparseMatrixRef::new(&indptr, &indices, &values, 1, 6);
    let stats = marker_statistics(smr, &[0, 1, 2], &[3, 4, 5], 1.0).unwrap();
    assert!((stats[0].auroc - 0.5).abs() < 1e-12, "auroc = {}", stats[0].auroc);
    assert!(stats[0].auroc_power() < 1e-12);
}

/// AUROC must equal the Mann-Whitney U statistic normalised by n1*n2.
#[test]
fn auroc_agrees_with_mann_whitney_u() {
    let indptr = vec![0usize, 8];
    let indices: Vec<usize> = (0..8).collect();
    let values = vec![1.0f64, 7.0, 3.0, 9.0, 2.0, 8.0, 4.0, 6.0];

    let g1 = vec![0, 1, 2, 3];
    let g2 = vec![4, 5, 6, 7];

    let smr = SparseMatrixRef::new(&indptr, &indices, &values, 1, 8);
    let stats = marker_statistics(smr, &g1, &g2, 1.0).unwrap();

    let x: Vec<f64> = g1.iter().map(|&i| values[i]).collect();
    let y: Vec<f64> = g2.iter().map(|&i| values[i]).collect();
    let mw = mann_whitney_optimized(&x, &y, Alternative::Greater);
    // Alternative::Greater reports U_x directly.
    let expected_auc = mw.statistic / (x.len() * y.len()) as f64;

    assert!(
        (stats[0].auroc - expected_auc).abs() < 1e-12,
        "auroc = {} vs U/(n1*n2) = {}",
        stats[0].auroc,
        expected_auc
    );
}

#[test]
fn auroc_treats_stored_zeros_as_zeros() {
    let g1 = vec![0, 1, 2];
    let g2 = vec![3, 4, 5];

    // Cell 0 stored explicitly as 0.0
    let indptr_e = vec![0usize, 5];
    let indices_e = vec![0usize, 1, 2, 4, 5];
    let values_e = vec![0.0f64, 2.0, 3.0, 8.0, 9.0];
    let explicit = marker_statistics(
        SparseMatrixRef::new(&indptr_e, &indices_e, &values_e, 1, 6), &g1, &g2, 1.0,
    ).unwrap();

    // Cell 0 absent
    let indptr_i = vec![0usize, 4];
    let indices_i = vec![1usize, 2, 4, 5];
    let values_i = vec![2.0f64, 3.0, 8.0, 9.0];
    let implicit = marker_statistics(
        SparseMatrixRef::new(&indptr_i, &indices_i, &values_i, 1, 6), &g1, &g2, 1.0,
    ).unwrap();

    assert_eq!(explicit[0], implicit[0], "stored zero must equal absent entry");
    assert!((explicit[0].pct_group1 - 2.0 / 3.0).abs() < 1e-12);
}

#[test]
fn marker_candidate_filter_applies_both_thresholds() {
    let indptr = vec![0usize, 3, 6, 9];
    let indices = vec![0usize, 1, 2, 3, 4, 5, 0, 1, 3];
    let values = vec![
        5.0f64, 5.0, 5.0, // gene 0: group1 only  -> high pct1, high log2fc
        5.0, 5.0, 5.0,    // gene 1: group2 only  -> high pct2, high (negative) log2fc
        0.1, 0.1, 0.1,    // gene 2: low, spread  -> low log2fc
    ];

    let smr = SparseMatrixRef::new(&indptr, &indices, &values, 3, 6);
    let stats = marker_statistics(smr, &[0, 1, 2], &[3, 4, 5], 1.0).unwrap();

    let kept = filter_marker_candidates(&stats, 0.5, 1.0);
    assert!(kept.contains(&0), "gene 0 is a clear group-1 marker");
    assert!(kept.contains(&1), "gene 1 is a clear group-2 marker");
    assert!(!kept.contains(&2), "gene 2 should fail the log2fc threshold");
}

// ----------------------------------------------------------------- negative binomial

#[test]
fn negative_binomial_detects_a_clear_difference() {
    let g1: Vec<f64> = vec![1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 2.0, 1.0];
    let g2: Vec<f64> = vec![50.0, 60.0, 55.0, 52.0, 58.0, 51.0, 57.0, 54.0];

    let res = negative_binomial_test(&g1, &g2, Alternative::TwoSided);
    assert!(res.p_value < 0.01, "p = {} for a 25x difference", res.p_value);
    let log2fc = res.metadata.get("log2_fold_change").copied().unwrap();
    assert!(log2fc < 0.0, "group 1 is lower, so log2fc must be negative: {}", log2fc);
}

#[test]
fn negative_binomial_is_not_significant_for_identical_groups() {
    let g: Vec<f64> = vec![5.0, 6.0, 4.0, 5.0, 6.0, 4.0, 5.0, 5.0];
    let res = negative_binomial_test(&g, &g, Alternative::TwoSided);
    assert!(res.p_value > 0.9, "p = {} for identical groups", res.p_value);
    assert!(res.statistic.abs() < 1e-9, "z = {}", res.statistic);
}

#[test]
fn negative_binomial_estimates_nonnegative_dispersion() {
    // Overdispersed counts: variance well above the mean.
    let g1: Vec<f64> = vec![0.0, 1.0, 20.0, 0.0, 40.0, 2.0, 0.0, 35.0];
    let g2: Vec<f64> = vec![1.0, 0.0, 25.0, 3.0, 30.0, 0.0, 1.0, 28.0];

    let res = negative_binomial_test(&g1, &g2, Alternative::TwoSided);
    let alpha = res.metadata.get("dispersion").copied().unwrap();
    assert!(alpha > 0.0, "overdispersed data must give alpha > 0, got {}", alpha);

    // Under-dispersed (near-constant) counts must floor at zero, not go negative.
    let c: Vec<f64> = vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0];
    let res = negative_binomial_test(&c, &c, Alternative::TwoSided);
    let alpha = res.metadata.get("dispersion").copied().unwrap();
    assert!(alpha >= 0.0, "dispersion must never be negative, got {}", alpha);
}

#[test]
fn negative_binomial_handles_degenerate_input() {
    let tiny: Vec<f64> = vec![1.0];
    let res = negative_binomial_test(&tiny, &tiny, Alternative::TwoSided);
    assert_eq!(res.p_value, 1.0, "a single observation per group cannot be tested");

    let zeros: Vec<f64> = vec![0.0; 5];
    let res = negative_binomial_test(&zeros, &zeros, Alternative::TwoSided);
    assert_eq!(res.p_value, 1.0, "all-zero genes must not be called significant");
}
