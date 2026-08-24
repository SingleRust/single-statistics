//! Regression tests for previously-shipped correctness bugs.
//!
//! Each test here corresponds to a defect that was live in the crate and is
//! written to fail against the old behaviour.

use single_statistics::testing::correction::holm_bonferroni_correction;
use single_statistics::testing::inference::nonparametric::{
    mann_whitney_optimized, mann_whitney_sparse,
};
use single_statistics::testing::inference::MatrixStatTests;
use single_statistics::testing::utils::SparseMatrixRef;
use single_statistics::testing::Alternative;

/// Holm-Bonferroni is a step-down procedure and must produce monotone output.
/// Previously the running maximum was missing, so the largest p-value could
/// receive a smaller adjusted value than the one below it.
#[test]
fn holm_bonferroni_enforces_monotonicity() {
    let adjusted = holm_bonferroni_correction(&[0.01f64, 0.02, 0.03]).unwrap();
    // R: p.adjust(c(0.01, 0.02, 0.03), method = "holm") -> 0.03 0.04 0.04
    assert!((adjusted[0] - 0.03).abs() < 1e-12, "got {:?}", adjusted);
    assert!((adjusted[1] - 0.04).abs() < 1e-12, "got {:?}", adjusted);
    assert!((adjusted[2] - 0.04).abs() < 1e-12, "got {:?}", adjusted);
    assert!(adjusted[2] >= adjusted[1], "non-monotone: {:?}", adjusted);
}

/// A CSR matrix may store explicit zeros. Mann-Whitney previously counted such
/// an entry as neither a zero nor a non-zero, dropping it from the sample size
/// and corrupting the rank sums.
#[test]
fn mann_whitney_handles_explicitly_stored_zeros() {
    // Cell 0 holds an explicitly stored 0.0.
    let indptr = vec![0usize, 10];
    let indices: Vec<usize> = (0..10).collect();
    let values = vec![0.0f64, 1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0, 14.0];

    let smr = SparseMatrixRef::new(&indptr, &indices, &values, 1, 10);
    let g1: Vec<usize> = (0..5).collect();
    let g2: Vec<usize> = (5..10).collect();

    let sparse = mann_whitney_sparse(smr, &g1, &g2, Alternative::TwoSided).unwrap();
    // The dense path over the identical data is the reference.
    let dense = mann_whitney_optimized(
        &[0.0, 1.0, 2.0, 3.0, 4.0],
        &[10.0, 11.0, 12.0, 13.0, 14.0],
        Alternative::TwoSided,
    );

    assert!(
        (sparse[0].p_value - dense.p_value).abs() < 1e-12,
        "sparse path lost the stored zero: sparse p={} vs dense p={}",
        sparse[0].p_value,
        dense.p_value
    );
    assert!((sparse[0].statistic - dense.statistic).abs() < 1e-12);
}

/// An implicit zero and an explicitly stored zero must produce identical results.
#[test]
fn mann_whitney_stored_and_implicit_zeros_agree() {
    let g1: Vec<usize> = (0..5).collect();
    let g2: Vec<usize> = (5..10).collect();

    // Explicit: cell 0 stored as 0.0
    let indptr_e = vec![0usize, 10];
    let indices_e: Vec<usize> = (0..10).collect();
    let values_e = vec![0.0f64, 1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0, 14.0];
    let explicit = mann_whitney_sparse(
        SparseMatrixRef::new(&indptr_e, &indices_e, &values_e, 1, 10),
        &g1, &g2, Alternative::TwoSided,
    ).unwrap();

    // Implicit: cell 0 simply absent
    let indptr_i = vec![0usize, 9];
    let indices_i: Vec<usize> = (1..10).collect();
    let values_i = vec![1.0f64, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0, 14.0];
    let implicit = mann_whitney_sparse(
        SparseMatrixRef::new(&indptr_i, &indices_i, &values_i, 1, 10),
        &g1, &g2, Alternative::TwoSided,
    ).unwrap();

    assert!(
        (explicit[0].p_value - implicit[0].p_value).abs() < 1e-12,
        "explicit zero p={} != implicit zero p={}",
        explicit[0].p_value, implicit[0].p_value
    );
}

/// Fisher's exact test previously counted every *stored* entry as "expressed",
/// never inspecting the value, so explicit zeros collapsed the contingency table.
#[test]
fn fisher_exact_ignores_explicitly_stored_zeros() {
    // Group 1 cells (0, 1) hold stored 0.0 -> not expressed.
    // Group 2 cells (2, 3) hold 1.0       -> expressed.
    // Truth: a = 0, b = 2, c = 2, d = 0.
    let indptr = vec![0usize, 4];
    let indices = vec![0usize, 1, 2, 3];
    let values = vec![0.0f64, 0.0, 1.0, 1.0];

    let smr = SparseMatrixRef::new(&indptr, &indices, &values, 1, 4);
    let res = smr
        .fisher_exact_test(&[0, 1], &[2, 3], Alternative::TwoSided)
        .unwrap();

    assert!(
        res[0].p_value < 1.0,
        "stored zeros counted as expressed; table collapsed to a=2,b=2 (p={})",
        res[0].p_value
    );
}

/// A stored zero and an absent entry must be indistinguishable to Fisher too.
#[test]
fn fisher_exact_stored_and_implicit_zeros_agree() {
    let indptr_e = vec![0usize, 4];
    let indices_e = vec![0usize, 1, 2, 3];
    let values_e = vec![0.0f64, 0.0, 1.0, 1.0];
    let explicit = SparseMatrixRef::new(&indptr_e, &indices_e, &values_e, 1, 4)
        .fisher_exact_test(&[0, 1], &[2, 3], Alternative::TwoSided)
        .unwrap();

    let indptr_i = vec![0usize, 2];
    let indices_i = vec![2usize, 3];
    let values_i = vec![1.0f64, 1.0];
    let implicit = SparseMatrixRef::new(&indptr_i, &indices_i, &values_i, 1, 4)
        .fisher_exact_test(&[0, 1], &[2, 3], Alternative::TwoSided)
        .unwrap();

    assert!(
        (explicit[0].p_value - implicit[0].p_value).abs() < 1e-12,
        "explicit p={} != implicit p={}",
        explicit[0].p_value, implicit[0].p_value
    );
}

/// The large-df t-test tail used a hand-rolled Chebyshev erfc that mixed
/// Abramowitz & Stegun 7.1.28's coefficients into 7.1.26's functional form,
/// making every p-value roughly 4x too small for |t| >= 2.83 whenever df > 100.
#[test]
fn large_df_t_test_tail_is_accurate() {
    use single_statistics::testing::inference::parametric::fast_t_test_from_sums;
    use single_statistics::testing::TTestType;

    // n = 1000 per group, unit variance, mean difference tuned to give t = 3.
    let n = 1000.0;
    let mean_diff = 3.0 * (2.0f64 / n).sqrt();
    let sum1 = mean_diff * n;
    let sum_sq1 = (n - 1.0) + sum1 * sum1 / n;
    let sum_sq2 = n - 1.0;

    let r = fast_t_test_from_sums(sum1, sum_sq1, n, 0.0, sum_sq2, n, TTestType::Welch);

    assert!((r.statistic - 3.0).abs() < 1e-9, "t = {}", r.statistic);
    assert!(r.degrees_of_freedom.unwrap() > 100.0);

    // Two-sided normal tail at |z| = 3.
    let expected = 2.0 * 0.001_349_898_031_630_1;
    assert!(
        (r.p_value - expected).abs() < 1e-9,
        "p = {} (expected {}); the old code gave ~6.5e-4",
        r.p_value, expected
    );
}
