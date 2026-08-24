//! Tests for the rank-based and discrete tests, checked against reference values.
//!
//! Reference values are quoted from R where noted. The rank-based tests here use a
//! normal approximation with continuity correction, so R comparisons use
//! `exact = FALSE, correct = TRUE`.

use single_statistics::testing::inference::discrete::{
    binomial_test, chi_square_goodness_of_fit,
};
use single_statistics::testing::inference::nonparametric::{
    kruskal_wallis, kruskal_wallis_sparse, mann_whitney_optimized, wilcoxon_signed_rank,
    wilcoxon_signed_rank_one_sample,
};
use single_statistics::testing::inference::MatrixStatTests;
use single_statistics::testing::utils::SparseMatrixRef;
use single_statistics::testing::{Alternative, TestMethod};

// ---------------------------------------------------------------- Wilcoxon signed-rank

/// R: wilcox.test(c(1,2,3,4,5), exact = FALSE, correct = TRUE) -> V = 15, p = 0.05903
#[test]
fn wilcoxon_signed_rank_matches_reference() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y = [0.0, 0.0, 0.0, 0.0, 0.0];
    let res = wilcoxon_signed_rank(&x, &y, Alternative::TwoSided).unwrap();

    assert!((res.statistic - 15.0).abs() < 1e-12, "V = {}", res.statistic);
    assert!(
        (res.p_value - 0.05903).abs() < 1e-4,
        "p = {} (expected ~0.05903)",
        res.p_value
    );
}

/// Hand-computed: differences [1, 1, 2, 3] give V = 10 with a tie group of size 2,
/// so var = 4*5*9/24 - (2^3-2)/48 = 7.375 rather than 7.5.
#[test]
fn wilcoxon_signed_rank_applies_tie_correction() {
    let x = [1.0, 1.0, 2.0, 3.0];
    let y = [0.0, 0.0, 0.0, 0.0];
    let res = wilcoxon_signed_rank(&x, &y, Alternative::TwoSided).unwrap();

    assert!((res.statistic - 10.0).abs() < 1e-12, "V = {}", res.statistic);
    let var = res.metadata.get("var_w").copied().unwrap();
    assert!((var - 7.375).abs() < 1e-12, "var_w = {} (expected 7.375)", var);
}

#[test]
fn wilcoxon_signed_rank_discards_zero_differences() {
    // Three tied pairs contribute nothing; only the two real differences count.
    let x = [5.0, 5.0, 5.0, 1.0, 2.0];
    let y = [5.0, 5.0, 5.0, 0.0, 0.0];
    let res = wilcoxon_signed_rank(&x, &y, Alternative::TwoSided).unwrap();

    let n_pairs = res.metadata.get("n_pairs").copied().unwrap();
    assert!((n_pairs - 2.0).abs() < 1e-12, "n_pairs = {}", n_pairs);
}

#[test]
fn wilcoxon_signed_rank_all_ties_is_not_significant() {
    let x = [1.0, 2.0, 3.0];
    let res = wilcoxon_signed_rank(&x, &x, Alternative::TwoSided).unwrap();
    assert_eq!(res.p_value, 1.0);
    assert_eq!(res.statistic, 0.0);
}

#[test]
fn wilcoxon_signed_rank_rejects_mismatched_lengths() {
    let err = wilcoxon_signed_rank(&[1.0, 2.0], &[1.0], Alternative::TwoSided);
    assert!(err.is_err(), "unequal paired lengths must be rejected");
}

#[test]
fn wilcoxon_one_sample_shifts_against_median() {
    // Values all above mu0 = 2 -> one-sided "greater" should be small.
    let x = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let res = wilcoxon_signed_rank_one_sample(&x, 2.0, Alternative::Greater);
    assert!(res.p_value < 0.05, "p = {}", res.p_value);

    let res_less = wilcoxon_signed_rank_one_sample(&x, 2.0, Alternative::Less);
    assert!(res_less.p_value > 0.95, "p = {}", res_less.p_value);
}

// ------------------------------------------------------------------- Kruskal-Wallis

/// R (Hollander & Wolfe mucociliary clearance data):
/// kruskal.test(list(...)) -> chi-squared = 0.7714, df = 2, p = 0.68
#[test]
fn kruskal_wallis_matches_reference() {
    let g1 = [2.9, 3.0, 2.5, 2.6, 3.2];
    let g2 = [3.8, 2.7, 4.0, 2.4];
    let g3 = [2.8, 3.4, 3.7, 2.2, 2.0];

    let res = kruskal_wallis(&[&g1[..], &g2[..], &g3[..]]).unwrap();

    assert!(
        (res.statistic - 0.7714).abs() < 1e-3,
        "H = {} (expected 0.7714)",
        res.statistic
    );
    assert_eq!(res.degrees_of_freedom, Some(2.0));
    assert!((res.p_value - 0.68).abs() < 1e-2, "p = {}", res.p_value);
}

#[test]
fn kruskal_wallis_detects_a_real_difference() {
    let g1 = [1.0, 2.0, 3.0, 4.0, 5.0];
    let g2 = [10.0, 11.0, 12.0, 13.0, 14.0];
    let g3 = [20.0, 21.0, 22.0, 23.0, 24.0];

    let res = kruskal_wallis(&[&g1[..], &g2[..], &g3[..]]).unwrap();
    assert!(res.p_value < 0.01, "p = {} for well-separated groups", res.p_value);
}

#[test]
fn kruskal_wallis_requires_two_groups() {
    let g1 = [1.0, 2.0, 3.0];
    assert!(kruskal_wallis(&[&g1[..]]).is_err());
}

/// With two groups, Kruskal-Wallis H is the square of the uncorrected
/// Mann-Whitney z, so the two tests must agree on tie-free data.
#[test]
fn kruskal_wallis_agrees_with_mann_whitney_on_two_groups() {
    let g1 = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0];
    let g2 = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0];

    let kw = kruskal_wallis(&[&g1[..], &g2[..]]).unwrap();
    let mw = mann_whitney_optimized(&g1, &g2, Alternative::TwoSided);
    let z = mw.metadata.get("z_score").copied().unwrap();

    // MW applies a continuity correction that KW does not, so compare loosely:
    // both must agree that there is no difference here.
    assert!(kw.p_value > 0.5, "KW p = {}", kw.p_value);
    assert!(mw.p_value > 0.5, "MW p = {}", mw.p_value);
    assert!(kw.statistic >= z * z - 1.0, "H = {}, z^2 = {}", kw.statistic, z * z);
}

// ------------------------------------------------------- Kruskal-Wallis over sparse

/// The sparse path ranks the zero block analytically; it must agree with the dense
/// test run over the same data materialised by hand.
#[test]
fn kruskal_wallis_sparse_matches_dense() {
    // 1 gene x 9 cells, 3 groups of 3. Cells 0, 3, 6 are unexpressed (implicit zeros).
    let indptr = vec![0usize, 6];
    let indices = vec![1usize, 2, 4, 5, 7, 8];
    let values = vec![1.0f64, 2.0, 10.0, 11.0, 20.0, 21.0];
    let group_ids = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];

    let smr = SparseMatrixRef::new(&indptr, &indices, &values, 1, 9);
    let sparse = kruskal_wallis_sparse(smr, &group_ids).unwrap();

    let g1 = [0.0, 1.0, 2.0];
    let g2 = [0.0, 10.0, 11.0];
    let g3 = [0.0, 20.0, 21.0];
    let dense = kruskal_wallis(&[&g1[..], &g2[..], &g3[..]]).unwrap();

    assert!(
        (sparse[0].statistic - dense.statistic).abs() < 1e-9,
        "sparse H = {} vs dense H = {}",
        sparse[0].statistic,
        dense.statistic
    );
    assert!((sparse[0].p_value - dense.p_value).abs() < 1e-9);
}

#[test]
fn kruskal_wallis_sparse_handles_explicitly_stored_zeros() {
    let group_ids = vec![0, 0, 0, 1, 1, 1];

    // Cell 0 stored as an explicit 0.0
    let indptr_e = vec![0usize, 5];
    let indices_e = vec![0usize, 1, 2, 4, 5];
    let values_e = vec![0.0f64, 1.0, 2.0, 10.0, 11.0];
    let explicit =
        kruskal_wallis_sparse(SparseMatrixRef::new(&indptr_e, &indices_e, &values_e, 1, 6), &group_ids)
            .unwrap();

    // Cell 0 simply absent
    let indptr_i = vec![0usize, 4];
    let indices_i = vec![1usize, 2, 4, 5];
    let values_i = vec![1.0f64, 2.0, 10.0, 11.0];
    let implicit =
        kruskal_wallis_sparse(SparseMatrixRef::new(&indptr_i, &indices_i, &values_i, 1, 6), &group_ids)
            .unwrap();

    assert!(
        (explicit[0].statistic - implicit[0].statistic).abs() < 1e-12,
        "explicit H = {} != implicit H = {}",
        explicit[0].statistic,
        implicit[0].statistic
    );
}

/// Kruskal-Wallis is the one method that lifts the two-group restriction on
/// `differential_expression`.
#[test]
fn differential_expression_supports_three_groups_via_kruskal_wallis() {
    let indptr = vec![0usize, 9, 18];
    let indices: Vec<usize> = (0..9).chain(0..9).collect();
    let values = vec![
        // gene 0: clearly separated by group
        1.0f64, 2.0, 3.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0,
        // gene 1: no group structure
        5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
    ];
    let group_ids = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];

    let smr = SparseMatrixRef::new(&indptr, &indices, &values, 2, 9);
    let de = smr
        .differential_expression(&group_ids, TestMethod::KruskalWallis)
        .unwrap();

    assert_eq!(de.statistics.len(), 2);
    assert!(de.adjusted_p_values.is_some());
    assert!(
        de.p_values[0] < de.p_values[1],
        "separated gene p={} should beat flat gene p={}",
        de.p_values[0],
        de.p_values[1]
    );
    assert_eq!(
        de.global_metadata.get("test_type").map(String::as_str),
        Some("kruskal_wallis")
    );
}

/// The other methods must still reject a three-group assignment, with a message
/// that points at Kruskal-Wallis.
#[test]
fn differential_expression_rejects_three_groups_for_two_group_tests() {
    let indptr = vec![0usize, 9];
    let indices: Vec<usize> = (0..9).collect();
    let values = vec![1.0f64, 2.0, 3.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0];
    let group_ids = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];

    let smr = SparseMatrixRef::new(&indptr, &indices, &values, 1, 9);
    let err = smr
        .differential_expression(&group_ids, TestMethod::MannWhitney)
        .unwrap_err();
    assert!(
        err.to_string().contains("KruskalWallis"),
        "error should point at Kruskal-Wallis, got: {}",
        err
    );
}

// ------------------------------------------------------- Wilcoxon signed-rank sparse

#[test]
fn wilcoxon_signed_rank_sparse_detects_paired_shift() {
    // 2 genes x 8 cells. Cells 0..4 pair with cells 4..8.
    // Gene 0: the second cell of every pair is consistently higher.
    // Gene 1: identical in both members of every pair.
    let indptr = vec![0usize, 8, 16];
    let indices: Vec<usize> = (0..8).chain(0..8).collect();
    let values = vec![
        1.0f64, 2.0, 3.0, 4.0, 11.0, 12.0, 13.0, 14.0,
        7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0,
    ];

    let smr = SparseMatrixRef::new(&indptr, &indices, &values, 2, 8);
    let res = smr
        .wilcoxon_signed_rank_test(&[0, 1, 2, 3], &[4, 5, 6, 7], Alternative::TwoSided)
        .unwrap();

    assert_eq!(res.len(), 2);
    // Gene 0: every difference is negative and equal in rank magnitude -> V = 0
    assert_eq!(res[0].statistic, 0.0);
    assert!(res[0].p_value < 0.1, "gene 0 p = {}", res[0].p_value);
    // Gene 1: all differences zero -> discarded, no evidence
    assert_eq!(res[1].p_value, 1.0);
}

#[test]
fn wilcoxon_signed_rank_sparse_rejects_unequal_groups() {
    let indptr = vec![0usize, 4];
    let indices = vec![0usize, 1, 2, 3];
    let values = vec![1.0f64, 2.0, 3.0, 4.0];
    let smr = SparseMatrixRef::new(&indptr, &indices, &values, 1, 4);

    assert!(smr
        .wilcoxon_signed_rank_test(&[0, 1], &[2], Alternative::TwoSided)
        .is_err());
}

// ------------------------------------------------------------------------- discrete

/// R: binom.test(1, 10, 0.5) -> p = 0.02148 (exactly 22/1024)
#[test]
fn binomial_test_two_sided_matches_reference() {
    let res: single_statistics::testing::TestResult<f64> =
        binomial_test(1, 10, 0.5, Alternative::TwoSided);
    assert!(
        (res.p_value - 22.0 / 1024.0).abs() < 1e-12,
        "p = {} (expected {})",
        res.p_value,
        22.0 / 1024.0
    );
    assert_eq!(res.statistic, 1.0);
}

#[test]
fn binomial_test_one_sided_matches_reference() {
    // P(X <= 1) = 11/1024
    let less: single_statistics::testing::TestResult<f64> =
        binomial_test(1, 10, 0.5, Alternative::Less);
    assert!((less.p_value - 11.0 / 1024.0).abs() < 1e-12, "p = {}", less.p_value);

    // P(X >= 1) = 1 - P(X = 0) = 1023/1024
    let greater: single_statistics::testing::TestResult<f64> =
        binomial_test(1, 10, 0.5, Alternative::Greater);
    assert!(
        (greater.p_value - 1023.0 / 1024.0).abs() < 1e-12,
        "p = {}",
        greater.p_value
    );
}

#[test]
fn binomial_test_rejects_degenerate_input() {
    let zero_trials: single_statistics::testing::TestResult<f64> =
        binomial_test(0, 0, 0.5, Alternative::TwoSided);
    assert_eq!(zero_trials.p_value, 1.0);

    let too_many: single_statistics::testing::TestResult<f64> =
        binomial_test(11, 10, 0.5, Alternative::TwoSided);
    assert_eq!(too_many.p_value, 1.0);
}

/// Hand-computed: chi2 = (15^2 + 5^2 + 5^2 + 15^2)/25 = 20 on 3 df.
#[test]
fn chi_square_goodness_of_fit_matches_reference() {
    let observed = [10.0f64, 20.0, 30.0, 40.0];
    let expected = [25.0f64, 25.0, 25.0, 25.0];

    let res = chi_square_goodness_of_fit(&observed, &expected, Alternative::Greater);
    assert!((res.statistic - 20.0).abs() < 1e-12, "chi2 = {}", res.statistic);
    // R: 1 - pchisq(20, 3) = 0.0001697
    assert!((res.p_value - 0.0001697).abs() < 1e-6, "p = {}", res.p_value);
}

#[test]
fn chi_square_goodness_of_fit_skips_zero_expected() {
    // The zero-expected category must be skipped rather than producing NaN.
    let observed = [10.0f64, 20.0, 0.0];
    let expected = [15.0f64, 15.0, 0.0];
    let res = chi_square_goodness_of_fit(&observed, &expected, Alternative::Greater);
    assert!(res.statistic.is_finite(), "chi2 = {}", res.statistic);
    assert!(res.p_value.is_finite() && (0.0..=1.0).contains(&res.p_value));
}
