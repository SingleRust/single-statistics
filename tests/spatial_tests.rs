//! Spatial autocorrelation. Reference values hand-computed in the comments.
#![cfg(feature = "spatial")]

use single_statistics::spatial::{gearys_c, morans_i, WeightMoments};
use single_statistics::testing::utils::SparseMatrixRef;

// -------------------------------------------------------------- spatial autocorrelation

/// A 4-spot chain graph: edges (0,1), (1,2), (2,3), symmetric, unit weights.
/// S0 = 6.
fn chain_weights() -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    // row 0 -> {1}, row 1 -> {0,2}, row 2 -> {1,3}, row 3 -> {2}
    let indptr = vec![0usize, 1, 3, 5, 6];
    let indices = vec![1usize, 0, 2, 1, 3, 2];
    let values = vec![1.0f64; 6];
    (indptr, indices, values)
}

#[test]
fn weight_moments_match_hand_calculation() {
    let (wp, wi, wv) = chain_weights();
    let w = SparseMatrixRef::new(&wp, &wi, &wv, 4, 4);
    let m = WeightMoments::from_weights(w).unwrap();

    assert_eq!(m.n, 4);
    assert!((m.s0 - 6.0).abs() < 1e-12, "S0 = {}", m.s0);
    // Symmetric unit weights: each of the 6 directed edges contributes (1+1)^2 = 4,
    // so S1 = 0.5 * 6 * 4 = 12.
    assert!((m.s1 - 12.0).abs() < 1e-12, "S1 = {}", m.s1);
    // Row+col sums are [2, 4, 4, 2] -> S2 = 4 + 16 + 16 + 4 = 40.
    assert!((m.s2 - 40.0).abs() < 1e-12, "S2 = {}", m.s2);
}

/// Hand-computed: x = [1,1,2,2] on the chain gives I = (4/6)*(0.5/1.0) = 1/3.
#[test]
fn morans_i_matches_hand_calculation() {
    let (wp, wi, wv) = chain_weights();
    let w = SparseMatrixRef::new(&wp, &wi, &wv, 4, 4);

    let indptr = vec![0usize, 4];
    let indices = vec![0usize, 1, 2, 3];
    let values = vec![1.0f64, 1.0, 2.0, 2.0];
    let x = SparseMatrixRef::new(&indptr, &indices, &values, 1, 4);

    let res = morans_i(x, w).unwrap();
    assert!(
        (res[0].statistic - 1.0 / 3.0).abs() < 1e-12,
        "I = {} (expected 0.3333)",
        res[0].statistic
    );
    // E[I] = -1/(n-1) = -1/3
    let expected = res[0].metadata.get("expected").copied().unwrap();
    assert!((expected + 1.0 / 3.0).abs() < 1e-12, "E[I] = {}", expected);
}

/// Hand-computed: the alternating pattern x = [1,2,1,2] gives I = (4/6)*(-1.5) = -1.
#[test]
fn morans_i_detects_negative_autocorrelation() {
    let (wp, wi, wv) = chain_weights();
    let w = SparseMatrixRef::new(&wp, &wi, &wv, 4, 4);

    let indptr = vec![0usize, 4];
    let indices = vec![0usize, 1, 2, 3];
    let values = vec![1.0f64, 2.0, 1.0, 2.0];
    let x = SparseMatrixRef::new(&indptr, &indices, &values, 1, 4);

    let res = morans_i(x, w).unwrap();
    assert!(
        (res[0].statistic + 1.0).abs() < 1e-12,
        "I = {} (expected -1.0)",
        res[0].statistic
    );
}

/// Hand-computed: x = [1,1,2,2] gives C = (3*2)/(2*6*1) = 0.5 (positive autocorrelation).
#[test]
fn gearys_c_matches_hand_calculation() {
    let (wp, wi, wv) = chain_weights();
    let w = SparseMatrixRef::new(&wp, &wi, &wv, 4, 4);

    let indptr = vec![0usize, 4];
    let indices = vec![0usize, 1, 2, 3];
    let values = vec![1.0f64, 1.0, 2.0, 2.0];
    let x = SparseMatrixRef::new(&indptr, &indices, &values, 1, 4);

    let res = gearys_c(x, w).unwrap();
    assert!(
        (res[0].statistic - 0.5).abs() < 1e-12,
        "C = {} (expected 0.5)",
        res[0].statistic
    );
    assert!(res[0].statistic < 1.0, "C < 1 means positive autocorrelation");
}

/// Hand-computed: x = [1,2,1,2] gives C = (3*6)/(2*6*1) = 1.5.
#[test]
fn gearys_c_detects_negative_autocorrelation() {
    let (wp, wi, wv) = chain_weights();
    let w = SparseMatrixRef::new(&wp, &wi, &wv, 4, 4);

    let indptr = vec![0usize, 4];
    let indices = vec![0usize, 1, 2, 3];
    let values = vec![1.0f64, 2.0, 1.0, 2.0];
    let x = SparseMatrixRef::new(&indptr, &indices, &values, 1, 4);

    let res = gearys_c(x, w).unwrap();
    assert!(
        (res[0].statistic - 1.5).abs() < 1e-12,
        "C = {} (expected 1.5)",
        res[0].statistic
    );
}

/// Moran's I and Geary's C must agree on the *direction* of spatial structure.
#[test]
fn morans_and_gearys_agree_on_direction() {
    let (wp, wi, wv) = chain_weights();

    let indptr = vec![0usize, 4, 8];
    let indices = vec![0usize, 1, 2, 3, 0, 1, 2, 3];
    let values = vec![
        1.0f64, 1.0, 2.0, 2.0, // clustered
        1.0, 2.0, 1.0, 2.0,    // alternating
    ];

    let i_res = morans_i(
        SparseMatrixRef::new(&indptr, &indices, &values, 2, 4),
        SparseMatrixRef::new(&wp, &wi, &wv, 4, 4),
    ).unwrap();
    let c_res = gearys_c(
        SparseMatrixRef::new(&indptr, &indices, &values, 2, 4),
        SparseMatrixRef::new(&wp, &wi, &wv, 4, 4),
    ).unwrap();

    // Clustered: I above its expectation, C below 1.
    assert!(i_res[0].statistic > -1.0 / 3.0 && c_res[0].statistic < 1.0);
    // Alternating: I below its expectation, C above 1.
    assert!(i_res[1].statistic < -1.0 / 3.0 && c_res[1].statistic > 1.0);
}

#[test]
fn spatial_rejects_shape_mismatch() {
    let (wp, wi, wv) = chain_weights();
    let w = SparseMatrixRef::new(&wp, &wi, &wv, 4, 4);

    // 5 spots in the expression matrix, 4 in the weights
    let indptr = vec![0usize, 5];
    let indices = vec![0usize, 1, 2, 3, 4];
    let values = vec![1.0f64; 5];
    let x = SparseMatrixRef::new(&indptr, &indices, &values, 1, 5);

    assert!(morans_i(x, w).is_err(), "shape mismatch must be rejected");
}

#[test]
fn spatial_handles_constant_gene() {
    let (wp, wi, wv) = chain_weights();
    let w = SparseMatrixRef::new(&wp, &wi, &wv, 4, 4);

    let indptr = vec![0usize, 4];
    let indices = vec![0usize, 1, 2, 3];
    let values = vec![3.0f64; 4];
    let x = SparseMatrixRef::new(&indptr, &indices, &values, 1, 4);

    let res = morans_i(x, w).unwrap();
    assert!(res[0].statistic.is_nan(), "a constant gene has no spatial signal");
    assert_eq!(res[0].p_value, 1.0);
}

