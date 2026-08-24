#![cfg(feature = "spatial")]
use single_statistics::spatial::{gearys_c, gearys_c_permutation, morans_i, morans_i_permutation};
use single_statistics::testing::inference::parametric::fast_t_test_from_sums;
use single_statistics::testing::utils::SparseMatrixRef;
use single_statistics::testing::{MatrixStatTests, TTestType, TestMethod};

// ------------------------------------------------------------------ confidence intervals

/// x = [1,2,3] (mean 2), y = [5,6,7] (mean 6), n=3 each, pooled var 1.
/// mean_diff = -4, se = sqrt(1 * (1/3 + 1/3)) = 0.8165, df = 4, t_.975(4) = 2.7764
/// CI = -4 +/- 2.2678
#[test]
fn t_test_ci_matches_hand_calculation() {
    let r = fast_t_test_from_sums(6.0, 14.0, 3.0, 18.0, 110.0, 3.0, TTestType::Student);
    let (lo, hi) = r.confidence_interval.expect("CI must be populated");

    let se = (2.0f64 / 3.0).sqrt();
    let expected = 2.776_445_105_198_5 * se;
    assert!((lo - (-4.0 - expected)).abs() < 1e-6, "lo = {}", lo);
    assert!((hi - (-4.0 + expected)).abs() < 1e-6, "hi = {}", hi);
    assert_eq!(r.degrees_of_freedom, Some(4.0));
}

#[test]
fn t_test_ci_brackets_the_difference_and_excludes_zero_when_significant() {
    let r = fast_t_test_from_sums(6.0, 14.0, 3.0, 18.0, 110.0, 3.0, TTestType::Student);
    let (lo, hi) = r.confidence_interval.unwrap();
    assert!(lo < -4.0 && hi > -4.0, "CI must bracket the difference");
    assert!(hi < 0.0, "significant result -> CI excludes 0 (p = {})", r.p_value);
}

#[test]
fn t_test_ci_includes_zero_when_not_significant() {
    // identical groups
    let r = fast_t_test_from_sums(6.0, 14.0, 3.0, 6.0, 14.0, 3.0, TTestType::Student);
    let (lo, hi) = r.confidence_interval.unwrap();
    assert!(lo <= 0.0 && hi >= 0.0, "CI = ({}, {})", lo, hi);
}

#[test]
fn degenerate_input_gives_no_usable_ci() {
    let r = fast_t_test_from_sums(0.0, 0.0, 3.0, 0.0, 0.0, 3.0, TTestType::Student);
    let (lo, hi) = r.confidence_interval.unwrap();
    assert!(lo.is_nan() && hi.is_nan(), "zero variance -> NaN CI, got ({}, {})", lo, hi);
}

// ------------------------------------------------- the previously-dead result fields

#[test]
fn differential_expression_populates_intervals_and_metadata() {
    let indptr = vec![0usize, 6];
    let indices: Vec<usize> = (0..6).collect();
    let values = vec![1.0f64, 2.0, 3.0, 5.0, 6.0, 7.0];
    let smr = SparseMatrixRef::new(&indptr, &indices, &values, 1, 6);

    let de = smr
        .differential_expression(&[0, 0, 0, 1, 1, 1], TestMethod::TTest(TTestType::Welch))
        .unwrap();

    let cis = de.confidence_intervals.expect("t-test should carry CIs");
    assert_eq!(cis.len(), 1);
    assert!(cis[0].0 < cis[0].1, "CI = {:?}", cis[0]);

    let meta = de.feature_metadata.expect("per-gene metadata should be filled");
    assert_eq!(meta.len(), 1);
}

#[test]
fn feature_metadata_carries_per_test_fields() {
    let indptr = vec![0usize, 6];
    let indices: Vec<usize> = (0..6).collect();
    let values = vec![1.0f64, 2.0, 3.0, 5.0, 6.0, 7.0];
    let smr = SparseMatrixRef::new(&indptr, &indices, &values, 1, 6);

    let de = smr
        .differential_expression(&[0, 0, 0, 1, 1, 1], TestMethod::MannWhitney)
        .unwrap();

    let meta = de.feature_metadata.unwrap();
    assert!(meta[0].contains_key("z_score"), "keys: {:?}", meta[0].keys());
    assert!(meta[0].contains_key("var_u"));
}

// ------------------------------------------------------------------ spatial permutation

fn chain_weights() -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    (vec![0usize, 1, 3, 5, 6], vec![1usize, 0, 2, 1, 3, 2], vec![1.0f64; 6])
}

/// A bigger chain so the permutation null has room to be informative.
fn long_chain(n: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut indptr = vec![0usize];
    let mut indices = Vec::new();
    for i in 0..n {
        if i > 0 { indices.push(i - 1); }
        if i + 1 < n { indices.push(i + 1); }
        indptr.push(indices.len());
    }
    let values = vec![1.0f64; indices.len()];
    (indptr, indices, values)
}

#[test]
fn permutation_gives_the_same_statistic_as_the_analytic_path() {
    let (wp, wi, wv) = chain_weights();
    let indptr = vec![0usize, 4];
    let indices = vec![0usize, 1, 2, 3];
    let values = vec![1.0f64, 1.0, 2.0, 2.0];

    let a = morans_i(
        SparseMatrixRef::new(&indptr, &indices, &values, 1, 4),
        SparseMatrixRef::new(&wp, &wi, &wv, 4, 4),
    ).unwrap();
    let b = morans_i_permutation(
        SparseMatrixRef::new(&indptr, &indices, &values, 1, 4),
        SparseMatrixRef::new(&wp, &wi, &wv, 4, 4),
        200, 42,
    ).unwrap();

    // Only the p-value machinery differs; I itself must be identical.
    assert!((a[0].statistic - b[0].statistic).abs() < 1e-12);
}

#[test]
fn permutation_detects_clustering() {
    let n = 60;
    let (wp, wi, wv) = long_chain(n);
    // First half low, second half high: strong spatial structure.
    let indptr = vec![0usize, n];
    let indices: Vec<usize> = (0..n).collect();
    let values: Vec<f64> = (0..n).map(|i| if i < n / 2 { 1.0 } else { 9.0 }).collect();

    let r = morans_i_permutation(
        SparseMatrixRef::new(&indptr, &indices, &values, 1, n),
        SparseMatrixRef::new(&wp, &wi, &wv, n, n),
        500, 7,
    ).unwrap();

    assert!(r[0].statistic > 0.5, "I = {}", r[0].statistic);
    assert!(r[0].p_value < 0.01, "p = {}", r[0].p_value);
}

#[test]
fn permutation_finds_nothing_in_noise() {
    let n = 60;
    let (wp, wi, wv) = long_chain(n);
    // Deterministic but spatially unstructured.
    let indptr = vec![0usize, n];
    let indices: Vec<usize> = (0..n).collect();
    let values: Vec<f64> = (0..n).map(|i| ((i * 37 + 11) % 17) as f64).collect();

    let r = morans_i_permutation(
        SparseMatrixRef::new(&indptr, &indices, &values, 1, n),
        SparseMatrixRef::new(&wp, &wi, &wv, n, n),
        500, 7,
    ).unwrap();

    assert!(r[0].p_value > 0.05, "p = {} for unstructured data", r[0].p_value);
}

#[test]
fn gearys_permutation_agrees_with_morans_on_direction() {
    let n = 60;
    let (wp, wi, wv) = long_chain(n);
    let indptr = vec![0usize, n];
    let indices: Vec<usize> = (0..n).collect();
    let values: Vec<f64> = (0..n).map(|i| if i < n / 2 { 1.0 } else { 9.0 }).collect();

    let c = gearys_c_permutation(
        SparseMatrixRef::new(&indptr, &indices, &values, 1, n),
        SparseMatrixRef::new(&wp, &wi, &wv, n, n),
        500, 7,
    ).unwrap();

    assert!(c[0].statistic < 1.0, "C = {} (clustering)", c[0].statistic);
    assert!(c[0].p_value < 0.01, "p = {}", c[0].p_value);

    // and the analytic path still gives the same C
    let analytic = gearys_c(
        SparseMatrixRef::new(&indptr, &indices, &values, 1, n),
        SparseMatrixRef::new(&wp, &wi, &wv, n, n),
    ).unwrap();
    assert!((c[0].statistic - analytic[0].statistic).abs() < 1e-12);
}

#[test]
fn permutation_is_reproducible() {
    let n = 40;
    let (wp, wi, wv) = long_chain(n);
    let indptr = vec![0usize, n];
    let indices: Vec<usize> = (0..n).collect();
    let values: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();

    let go = || morans_i_permutation(
        SparseMatrixRef::new(&indptr, &indices, &values, 1, n),
        SparseMatrixRef::new(&wp, &wi, &wv, n, n),
        100, 99,
    ).unwrap();

    assert_eq!(go()[0].p_value, go()[0].p_value);
}
