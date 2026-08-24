use single_statistics::testing::utils::SparseMatrixRef;
use single_statistics::testing::inference::MatrixStatTests;
use single_statistics::testing::{TTestType, Alternative, TestMethod};
use sprs::{CsMat, TriMat};

#[test]
fn test_sparse_matrix_ref_ttest() {
    // 2x4 (2 genes, 4 cells)
    // Gene 0: [1, 2, 10, 11]
    // Gene 1: [5, 5, 5, 5]
    
    let indptr = vec![0usize, 4, 8];
    let indices = vec![0usize, 1, 2, 3, 0, 1, 2, 3];
    let values = vec![1.0f64, 2.0, 10.0, 11.0, 5.0, 5.0, 5.0, 5.0];
    
    let matrix = SparseMatrixRef::new(&indptr, &indices, &values, 2, 4);
    
    let group1 = vec![0, 1];
    let group2 = vec![2, 3];
    
    let results = matrix.t_test(&group1, &group2, TTestType::Welch).unwrap();
    assert_eq!(results.len(), 2);
    assert!(results[0].p_value < 0.05);
    assert!(results[1].p_value > 0.9);
}

#[test]
fn test_sparse_matrix_ref_mann_whitney() {
    let indptr = vec![0u32, 4, 8];
    let indices = vec![0u32, 1, 2, 3, 0, 1, 2, 3];
    let values = vec![1.0f64, 2.0, 10.0, 11.0, 5.0, 5.0, 5.0, 5.0];
    
    let matrix = SparseMatrixRef::new(&indptr, &indices, &values, 2, 4);
    
    let group1 = vec![0, 1];
    let group2 = vec![2, 3];
    
    let results = matrix.mann_whitney_test(&group1, &group2, Alternative::TwoSided).unwrap();
    
    assert_eq!(results.len(), 2);
    assert!(results[0].statistic == 0.0 || results[0].statistic == 4.0);
    assert_eq!(results[1].p_value, 1.0);
}

#[test]
fn test_differential_expression_wrapper() {
    let indptr = vec![0usize, 4, 8];
    let indices = vec![0usize, 1, 2, 3, 0, 1, 2, 3];
    let values = vec![1.0f64, 2.0, 10.0, 11.0, 5.0, 5.0, 5.0, 5.0];
    
    let matrix = SparseMatrixRef::new(&indptr, &indices, &values, 2, 4);
    
    let group_ids = vec![0, 0, 1, 1];
    
    let de_results = matrix.differential_expression(&group_ids, TestMethod::TTest(TTestType::Welch)).unwrap();
    
    assert_eq!(de_results.statistics.len(), 2);
    assert!(de_results.statistics[0].abs() > 5.0);
}

#[test]
fn test_csr_matrix_integration() {
    let mut tri = TriMat::new((2, 4));
    tri.add_triplet(0, 0, 1.0); tri.add_triplet(0, 1, 2.0);
    tri.add_triplet(0, 2, 10.0); tri.add_triplet(0, 3, 11.0);
    tri.add_triplet(1, 0, 5.0); tri.add_triplet(1, 1, 5.0);
    tri.add_triplet(1, 2, 5.0); tri.add_triplet(1, 3, 5.0);
    let csr: CsMat<f64> = tri.to_csr();

    let group1 = vec![0, 1];
    let group2 = vec![2, 3];

    let results = csr.t_test(&group1, &group2, TTestType::Welch).unwrap();
    assert_eq!(results.len(), 2);
    assert!(results[0].p_value < 0.05);
}

/// The same data in CSC storage of the transposed shape must give identical
/// results, since the major axis is the feature axis either way.
#[test]
fn test_csc_matrix_matches_csr() {
    let mut tri = TriMat::new((2, 4));
    tri.add_triplet(0, 0, 1.0); tri.add_triplet(0, 1, 2.0);
    tri.add_triplet(0, 2, 10.0); tri.add_triplet(0, 3, 11.0);
    tri.add_triplet(1, 0, 5.0); tri.add_triplet(1, 1, 5.0);
    tri.add_triplet(1, 2, 5.0); tri.add_triplet(1, 3, 5.0);
    let csr: CsMat<f64> = tri.to_csr();

    // cells x genes in CSC -> major axis is still genes
    let mut tri_t = TriMat::new((4, 2));
    tri_t.add_triplet(0, 0, 1.0); tri_t.add_triplet(1, 0, 2.0);
    tri_t.add_triplet(2, 0, 10.0); tri_t.add_triplet(3, 0, 11.0);
    tri_t.add_triplet(0, 1, 5.0); tri_t.add_triplet(1, 1, 5.0);
    tri_t.add_triplet(2, 1, 5.0); tri_t.add_triplet(3, 1, 5.0);
    let csc: CsMat<f64> = tri_t.to_csc();

    let group1 = vec![0, 1];
    let group2 = vec![2, 3];

    let from_csr = csr.t_test(&group1, &group2, TTestType::Welch).unwrap();
    let from_csc = csc.t_test(&group1, &group2, TTestType::Welch).unwrap();

    assert_eq!(from_csr.len(), from_csc.len());
    for (a, b) in from_csr.iter().zip(from_csc.iter()) {
        // Gene 1 is constant, so its t-statistic is NaN in both paths; NaN != NaN,
        // so compare bit-for-bit agreement on nan-ness first.
        assert_eq!(
            a.statistic.is_nan(), b.statistic.is_nan(),
            "CSR t={} vs CSC t={}", a.statistic, b.statistic
        );
        if !a.statistic.is_nan() {
            assert!(
                (a.statistic - b.statistic).abs() < 1e-12,
                "CSR t={} vs CSC t={}", a.statistic, b.statistic
            );
        }
    }
}

/// An outer-sliced view keeps absolute indptr offsets while its indices/data are
/// sliced. `SprsView` rebases the offsets, so such a view must give the same answer
/// as the equivalent standalone matrix.
#[test]
fn test_outer_sliced_view_is_rebased_correctly() {
    // 4 genes x 4 cells, every entry stored.
    let mut tri = TriMat::new((4, 4));
    for i in 0..4 {
        for j in 0..4 {
            tri.add_triplet(i, j, (i * 4 + j) as f64 + 1.0);
        }
    }
    let csr: CsMat<f64> = tri.to_csr();

    let view = csr.view();
    let sliced = view.slice_outer(1..3); // genes 1 and 2
    assert!(!sliced.indptr().is_proper(), "test needs a non-proper view");

    let from_sliced = sliced
        .t_test(&[0, 1], &[2, 3], TTestType::Welch)
        .expect("a sliced view must be usable");

    // The same two genes, built standalone.
    let mut tri_ref = TriMat::new((2, 4));
    for (out_i, i) in (1..3).enumerate() {
        for j in 0..4 {
            tri_ref.add_triplet(out_i, j, (i * 4 + j) as f64 + 1.0);
        }
    }
    let reference: CsMat<f64> = tri_ref.to_csr();
    let from_reference = reference.t_test(&[0, 1], &[2, 3], TTestType::Welch).unwrap();

    assert_eq!(from_sliced.len(), 2);
    assert_eq!(from_sliced.len(), from_reference.len());
    for (a, b) in from_sliced.iter().zip(from_reference.iter()) {
        assert!(
            (a.statistic - b.statistic).abs() < 1e-12,
            "sliced t={} vs standalone t={}", a.statistic, b.statistic
        );
        assert!((a.p_value - b.p_value).abs() < 1e-12);
    }
}

/// The zero-copy fast path must stay zero-copy for ordinary matrices.
#[test]
fn test_unsliced_matrix_is_not_rebased() {
    use single_statistics::testing::utils::SprsView;

    let mut tri = TriMat::new((2, 4));
    tri.add_triplet(0, 0, 1.0); tri.add_triplet(0, 1, 2.0);
    tri.add_triplet(1, 2, 3.0); tri.add_triplet(1, 3, 4.0);
    let csr: CsMat<f64> = tri.to_csr();

    let view = SprsView::new(&csr);
    assert!(!view.is_rebased(), "an unsliced matrix must be borrowed, not copied");
    assert_eq!(view.n_features(), 2);
    assert_eq!(view.n_observations(), 4);
}

#[test]
fn test_sparse_matrix_ref_edge_cases() {
    use single_statistics::testing::utils::SparseMatrixRef;
    
    // 1. All-zero row (Gene 0 is empty)
    // 2. Identical values (Gene 1)
    let indptr = vec![0usize, 0, 4];
    let indices = vec![0usize, 1, 2, 3];
    let values = vec![1.0f64, 1.0, 1.0, 1.0];
    
    let smr = SparseMatrixRef::new(&indptr, &indices, &values, 2, 4);
    
    let group1 = vec![0, 1];
    let group2 = vec![2, 3];
    
    // T-test
    let results = smr.t_test(&group1, &group2, TTestType::Welch).unwrap();
    
    // Gene 0 (all zeros) should have NaN or handleable p-value
    assert!(results[0].p_value.is_nan() || results[0].p_value == 1.0);
    
    // Gene 1 (identical values)
    assert!(results[1].p_value.is_nan() || results[1].p_value == 1.0);
}

#[test]
fn test_sparse_matrix_ref_fisher() {
    use single_statistics::testing::inference::MatrixStatTests;
    use single_statistics::testing::Alternative;
    
    // Gene 0: [1, 1, 0, 0] - highly differential expression frequency
    // Gene 1: [1, 0, 1, 0] - no difference
    let indptr = vec![0usize, 2, 4];
    let indices = vec![0usize, 1, 0, 2];
    let values = vec![1.0f64, 1.0, 1.0, 1.0];
    
    let smr = SparseMatrixRef::new(&indptr, &indices, &values, 2, 4);
    
    let group1 = vec![0, 1];
    let group2 = vec![2, 3];
    
    let results = smr.fisher_exact_test(&group1, &group2, Alternative::TwoSided).unwrap();
    
    assert_eq!(results.len(), 2);
    // Gene 0 has (2,0) vs (0,2) expression. 
    // Group 1: 2 expr, 0 not expr
    // Group 2: 0 expr, 2 not expr
    // Fisher test should be significant (p=1/6 for 2-sided with small N=4 is often not <0.05, 
    // but it should be much smaller than Gene 1)
    assert!(results[0].p_value < results[1].p_value);
}
