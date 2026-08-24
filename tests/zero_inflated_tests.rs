use single_statistics::testing::inference::discrete::{zero_inflated_sparse, zero_inflated_test};
use single_statistics::testing::utils::SparseMatrixRef;
use single_statistics::testing::{Alternative, MatrixStatTests, TestMethod};

const TS: Alternative = Alternative::TwoSided;

/// Same expression where detected, but group 2 is mostly silent. Needs a realistic
/// group size: Fisher's exact on 6-vs-6 counts bottoms out around p = 0.06 and cannot
/// reach 0.05 however clean the dropout signal is.
#[test]
fn detects_a_dropout_difference() {
    let g1: Vec<f64> = (0..20).map(|i| 5.0 + (i % 3) as f64 * 0.5).collect();
    let mut g2 = vec![0.0; 16];
    g2.extend([5.0, 5.5, 6.0, 5.5]);

    let r = zero_inflated_test(&g1, &g2, TS);
    assert!(r.p_value < 0.001, "p = {}", r.p_value);

    let p_det: f64 = r.metadata.get("p_detection").copied().unwrap();
    let p_exp: f64 = r.metadata.get("p_expression").copied().unwrap();
    assert!(p_det < 0.001, "detection part should carry it: {}", p_det);
    assert!(p_exp > 0.05, "expression is unchanged where detected: {}", p_exp);
}

#[test]
fn detects_a_magnitude_difference() {
    // Detected everywhere, but group 2 is much higher.
    let g1 = vec![1.0, 1.2, 0.9, 1.1, 1.0, 1.05];
    let g2 = vec![9.0, 9.2, 8.8, 9.1, 9.0, 9.05];

    let r = zero_inflated_test(&g1, &g2, TS);
    assert!(r.p_value < 0.01, "p = {}", r.p_value);
    // Everything detected, so only the expression part is testable -> 2 df.
    assert_eq!(r.degrees_of_freedom, Some(2.0));
    let pd: f64 = r.metadata.get("p_detection").copied().unwrap();
    assert!(pd.is_nan());
}

#[test]
fn uses_both_parts_when_both_differ() {
    let g1 = vec![0.0, 0.0, 0.0, 1.0, 1.1, 0.9];
    let g2 = vec![9.0, 9.2, 8.8, 9.1, 9.0, 9.05];

    let r = zero_inflated_test(&g1, &g2, TS);
    assert_eq!(r.degrees_of_freedom, Some(4.0), "both parts should contribute");
    assert!(r.p_value < 0.01, "p = {}", r.p_value);
}

#[test]
fn identical_groups_are_not_significant() {
    let g = vec![0.0, 0.0, 5.0, 6.0, 5.5, 6.5];
    let r = zero_inflated_test(&g, &g, TS);
    assert!(r.p_value > 0.5, "p = {}", r.p_value);
}

#[test]
fn all_zero_gene_is_untestable() {
    let z = vec![0.0; 6];
    let r = zero_inflated_test(&z, &z, TS);
    assert_eq!(r.p_value, 1.0);
    assert_eq!(r.degrees_of_freedom, Some(0.0));
}

#[test]
fn reports_detection_rates() {
    let g1 = vec![0.0, 0.0, 1.0, 2.0]; // 50%
    let g2 = vec![1.0, 2.0, 3.0, 4.0]; // 100%
    let r = zero_inflated_test(&g1, &g2, TS);
    assert!((r.metadata.get("pct_group1").copied().unwrap() as f64 - 0.5).abs() < 1e-12);
    assert!((r.metadata.get("pct_group2").copied().unwrap() as f64 - 1.0).abs() < 1e-12);
}

#[test]
fn skips_expression_part_with_too_few_detections() {
    // Only one expressing cell in group 1 -> no variance, expression part dropped.
    let g1 = vec![0.0, 0.0, 0.0, 0.0, 0.0, 7.0];
    let g2 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let r = zero_inflated_test(&g1, &g2, TS);
    assert_eq!(r.degrees_of_freedom, Some(2.0));
    let pe: f64 = r.metadata.get("p_expression").copied().unwrap();
    assert!(pe.is_nan());
}

#[test]
fn handles_empty_groups() {
    let r = zero_inflated_test::<f64>(&[], &[1.0, 2.0], TS);
    assert_eq!(r.p_value, 1.0);
}

#[test]
fn sparse_path_matches_dense() {
    // 2 genes x 6 cells; gene 0 dropout-driven, gene 1 flat.
    let indptr = vec![0usize, 4, 10];
    let indices = vec![0usize, 1, 4, 5, 0, 1, 2, 3, 4, 5];
    let values = vec![5.0f64, 6.0, 5.5, 6.5, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0];
    let smr = SparseMatrixRef::new(&indptr, &indices, &values, 2, 6);

    let sparse = zero_inflated_sparse(smr, &[0, 1, 2], &[3, 4, 5], TS).unwrap();

    // gene 0 densely: cells 0,1 -> 5.0,6.0 ; cells 4,5 -> 5.5,6.5 ; 2,3 -> 0
    let dense = zero_inflated_test(&[5.0, 6.0, 0.0], &[0.0, 5.5, 6.5], TS);
    assert!((sparse[0].p_value - dense.p_value).abs() < 1e-12);
    assert_eq!(sparse.len(), 2);
}

#[test]
fn wired_into_differential_expression() {
    let indptr = vec![0usize, 4, 10];
    let indices = vec![0usize, 1, 2, 3, 0, 1, 2, 3, 4, 5];
    let values = vec![5.0f64, 6.0, 5.5, 6.5, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0];
    let smr = SparseMatrixRef::new(&indptr, &indices, &values, 2, 6);

    let de = smr
        .differential_expression(&[0, 0, 0, 1, 1, 1], TestMethod::ZeroInflated)
        .unwrap();

    assert_eq!(de.statistics.len(), 2);
    assert!(de.adjusted_p_values.is_some());
    assert_eq!(
        de.global_metadata.get("test_type").map(String::as_str),
        Some("zero_inflated")
    );
}
