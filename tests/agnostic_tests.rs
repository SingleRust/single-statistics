use single_statistics::testing::utils::SparseMatrixRef;
use single_statistics::testing::inference::MatrixStatTests;
use single_statistics::testing::{TTestType, Alternative, TestMethod};
use nalgebra_sparse::{CsrMatrix, CooMatrix};

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
    let mut coo = CooMatrix::new(2, 4);
    coo.push(0, 0, 1.0); coo.push(0, 1, 2.0); coo.push(0, 2, 10.0); coo.push(0, 3, 11.0);
    coo.push(1, 0, 5.0); coo.push(1, 1, 5.0); coo.push(1, 2, 5.0); coo.push(1, 3, 5.0);
    let csr = CsrMatrix::from(&coo);
    
    let group1 = vec![0, 1];
    let group2 = vec![2, 3];
    
    let results = csr.t_test(&group1, &group2, TTestType::Welch).unwrap();
    assert_eq!(results.len(), 2);
    assert!(results[0].p_value < 0.05);
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
