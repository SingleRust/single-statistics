//! Effect sizes. Reference values hand-computed in the comments.

use single_statistics::testing::effect::{
    calculate_cohens_d, calculate_hedges_g, calculate_log2_fold_change, cohens_d_from_samples,
};
use single_statistics::testing::utils::SparseMatrixRef;

/// 2 genes x 6 cells, fully stored.
/// gene 0: [1, 1, 1, 5, 5, 5]
/// gene 1: [4, 4, 4, 4, 4, 4]
fn fixture() -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let indptr = vec![0usize, 6, 12];
    let indices: Vec<usize> = (0..6).chain(0..6).collect();
    let values = vec![1.0f64, 1.0, 1.0, 5.0, 5.0, 5.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0];
    (indptr, indices, values)
}

// ------------------------------------------------------------------- log2 fold change

/// means 1 and 5, pseudocount 1 -> log2(2/6) = -1.58496
#[test]
fn log2fc_matches_hand_calculation() {
    let (p, i, v) = fixture();
    let m = SparseMatrixRef::new(&p, &i, &v, 2, 6);

    let fc = calculate_log2_fold_change(&m, 0, &[0, 1, 2], &[3, 4, 5], 1.0).unwrap();
    assert!((fc - (2.0f64 / 6.0).log2()).abs() < 1e-12, "log2fc = {}", fc);
}

/// Rows are genes: indices select cells along the minor axis. Asking for gene 1
/// (constant) must give 0 regardless of which cells are chosen.
#[test]
fn log2fc_indexes_cells_not_rows() {
    let (p, i, v) = fixture();
    let m = SparseMatrixRef::new(&p, &i, &v, 2, 6);

    let fc = calculate_log2_fold_change(&m, 1, &[0, 1, 2], &[3, 4, 5], 1.0).unwrap();
    assert!(fc.abs() < 1e-12, "constant gene must give 0, got {}", fc);

    // Swapping which cells form each group flips the sign on gene 0.
    let a = calculate_log2_fold_change(&m, 0, &[0, 1, 2], &[3, 4, 5], 1.0).unwrap();
    let b = calculate_log2_fold_change(&m, 0, &[3, 4, 5], &[0, 1, 2], 1.0).unwrap();
    assert!((a + b).abs() < 1e-12, "{} and {} should be negatives", a, b);
}

#[test]
fn log2fc_handles_an_unexpressed_group() {
    // gene 0 expressed only in cells 3,4,5
    let indptr = vec![0usize, 3];
    let indices = vec![3usize, 4, 5];
    let values = vec![5.0f64, 5.0, 5.0];
    let m = SparseMatrixRef::new(&indptr, &indices, &values, 1, 6);

    let fc = calculate_log2_fold_change(&m, 0, &[0, 1, 2], &[3, 4, 5], 1.0).unwrap();
    // means 0 and 5 -> log2(1/6)
    assert!(fc.is_finite(), "pseudocount must keep this finite, got {}", fc);
    assert!((fc - (1.0f64 / 6.0).log2()).abs() < 1e-12, "log2fc = {}", fc);
}

#[test]
fn log2fc_rejects_empty_groups() {
    let (p, i, v) = fixture();
    let m = SparseMatrixRef::new(&p, &i, &v, 2, 6);
    assert!(calculate_log2_fold_change(&m, 0, &[], &[3, 4, 5], 1.0).is_err());
    assert!(calculate_log2_fold_change(&m, 0, &[0, 1, 2], &[], 1.0).is_err());
}

// ------------------------------------------------------------------------- Cohen's d

/// Both groups constant -> pooled sd is 0 -> d is infinite, not a silent number.
#[test]
fn cohens_d_on_zero_variance_is_infinite() {
    let (p, i, v) = fixture();
    let m = SparseMatrixRef::new(&p, &i, &v, 2, 6);

    let d = calculate_cohens_d(&m, 0, &[0, 1, 2], &[3, 4, 5]).unwrap();
    assert!(d.is_infinite(), "d = {}", d);
}

/// group1 = [1,2,3] (mean 2, var 1), group2 = [5,6,7] (mean 6, var 1)
/// pooled sd = 1, d = (mean2 - mean1) / sd = 4
#[test]
fn cohens_d_matches_hand_calculation() {
    let indptr = vec![0usize, 6];
    let indices: Vec<usize> = (0..6).collect();
    let values = vec![1.0f64, 2.0, 3.0, 5.0, 6.0, 7.0];
    let m = SparseMatrixRef::new(&indptr, &indices, &values, 1, 6);

    let d = calculate_cohens_d(&m, 0, &[0, 1, 2], &[3, 4, 5]).unwrap();
    assert!((d - 4.0).abs() < 1e-12, "d = {}", d);
}

/// Unstored entries must be read as zeros, not skipped.
#[test]
fn cohens_d_materialises_implicit_zeros() {
    // gene 0 only expressed in cells 3,4,5 -> group1 is [0,0,0]
    let indptr = vec![0usize, 3];
    let indices = vec![3usize, 4, 5];
    let values = vec![5.0f64, 6.0, 7.0];
    let m = SparseMatrixRef::new(&indptr, &indices, &values, 1, 6);

    let d = calculate_cohens_d(&m, 0, &[0, 1, 2], &[3, 4, 5]).unwrap();
    // group1 = [0,0,0] var 0; group2 = [5,6,7] mean 6 var 1
    // pooled sd = sqrt((2*0 + 2*1)/4) = sqrt(0.5); d = 6 / sqrt(0.5)
    assert!((d - 6.0 / 0.5f64.sqrt()).abs() < 1e-12, "d = {}", d);
}

#[test]
fn cohens_d_needs_two_per_group() {
    let (p, i, v) = fixture();
    let m = SparseMatrixRef::new(&p, &i, &v, 2, 6);
    assert!(calculate_cohens_d(&m, 0, &[0], &[3, 4, 5]).is_err());
    assert!(calculate_cohens_d(&m, 0, &[0, 1, 2], &[3]).is_err());
}

/// The matrix path and the slice path must agree on the same numbers.
#[test]
fn cohens_d_matrix_and_sample_paths_agree() {
    let indptr = vec![0usize, 6];
    let indices: Vec<usize> = (0..6).collect();
    let values = vec![1.0f64, 2.0, 3.0, 5.0, 6.0, 7.0];
    let m = SparseMatrixRef::new(&indptr, &indices, &values, 1, 6);

    let from_matrix = calculate_cohens_d(&m, 0, &[0, 1, 2], &[3, 4, 5]).unwrap();
    let from_samples = cohens_d_from_samples(&[1.0, 2.0, 3.0], &[5.0, 6.0, 7.0]).unwrap();
    assert!((from_matrix - from_samples).abs() < 1e-12);
}

#[test]
fn cohens_d_from_samples_needs_two_per_group() {
    assert!(cohens_d_from_samples(&[1.0], &[5.0, 6.0]).is_err());
    assert!(cohens_d_from_samples::<f64>(&[], &[]).is_err());
}

// ------------------------------------------------------------------------- Hedges' g

/// g = J * d with J = 1 - 3/(4(n1+n2) - 9). For n1=n2=3, n=6:
/// J = 1 - 3/(4*4 - 1) = 1 - 3/15 = 0.8, and d = 4, so g = 3.2
#[test]
fn hedges_g_applies_the_correction_factor() {
    let indptr = vec![0usize, 6];
    let indices: Vec<usize> = (0..6).collect();
    let values = vec![1.0f64, 2.0, 3.0, 5.0, 6.0, 7.0];
    let m = SparseMatrixRef::new(&indptr, &indices, &values, 1, 6);

    let d = calculate_cohens_d(&m, 0, &[0, 1, 2], &[3, 4, 5]).unwrap();
    let g = calculate_hedges_g(&m, 0, &[0, 1, 2], &[3, 4, 5]).unwrap();

    let j = 1.0 - 3.0 / (4.0 * (6.0 - 2.0) - 1.0);
    assert!((g - j * d).abs() < 1e-12, "g = {}, expected {}", g, j * d);
    assert!(g.abs() < d.abs(), "g must shrink d");
}

/// The correction shrinks towards d as sample size grows.
#[test]
fn hedges_g_approaches_d_for_large_groups() {
    let n = 200usize;
    let indptr = vec![0usize, n];
    let indices: Vec<usize> = (0..n).collect();
    let values: Vec<f64> = (0..n).map(|i| if i < n / 2 { i as f64 } else { i as f64 + 50.0 }).collect();
    let m = SparseMatrixRef::new(&indptr, &indices, &values, 1, n);

    let g1: Vec<usize> = (0..n / 2).collect();
    let g2: Vec<usize> = (n / 2..n).collect();

    let d = calculate_cohens_d(&m, 0, &g1, &g2).unwrap();
    let g = calculate_hedges_g(&m, 0, &g1, &g2).unwrap();
    assert!((g / d - 1.0).abs() < 0.01, "g/d = {} should be near 1", g / d);
}

// ------------------------------------------------------------------ cross-check vs sprs

/// Effect sizes off an sprs matrix must match the raw-slice path.
#[test]
fn effect_sizes_agree_via_sprs() {
    use single_statistics::testing::utils::SprsView;
    use sprs::{CsMat, TriMat};

    let mut tri = TriMat::new((1, 6));
    for (c, v) in [1.0f64, 2.0, 3.0, 5.0, 6.0, 7.0].iter().enumerate() {
        tri.add_triplet(0, c, *v);
    }
    let csr: CsMat<f64> = tri.to_csr();
    let view = SprsView::new(&csr);
    let from_sprs = calculate_cohens_d(&view.as_matrix_ref(), 0, &[0, 1, 2], &[3, 4, 5]).unwrap();

    let indptr = vec![0usize, 6];
    let indices: Vec<usize> = (0..6).collect();
    let values = vec![1.0f64, 2.0, 3.0, 5.0, 6.0, 7.0];
    let raw = SparseMatrixRef::new(&indptr, &indices, &values, 1, 6);
    let from_raw = calculate_cohens_d(&raw, 0, &[0, 1, 2], &[3, 4, 5]).unwrap();

    assert!((from_sprs - from_raw).abs() < 1e-12);
}
