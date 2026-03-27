//! Non-parametric statistical tests for single-cell data analysis.
//!
//! This module implements non-parametric statistical tests that make fewer assumptions about
//! data distribution. These tests are particularly useful for single-cell data which often
//! exhibits non-normal distributions, high sparsity, and outliers.
//!
//! The primary test implemented is the Mann-Whitney U test (also known as the Wilcoxon 
//! rank-sum test), which compares the distributions of two groups without assuming normality.

use std::{cmp::Ordering, f64};

use nalgebra_sparse::CsrMatrix;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use single_utilities::traits::FloatOpsTS;
use statrs::distribution::{ContinuousCDF, Normal};

use crate::testing::{Alternative, TestResult};
use crate::testing::utils::SparseMatrixRef;
use num_traits::AsPrimitive;

/// Perform Mann-Whitney U tests on all genes comparing two groups of cells.
pub fn mann_whitney_matrix_groups<T>(
    matrix: &CsrMatrix<T>,
    group1_indices: &[usize],
    group2_indices: &[usize],
    alternative: Alternative,
) -> anyhow::Result<Vec<TestResult<f64>>>
where
    T: FloatOpsTS,
    f64: std::convert::From<T>,
{
    let smr = SparseMatrixRef {
        maj_ind: matrix.row_offsets(),
        min_ind: matrix.col_indices(),
        val: matrix.values(),
        n_rows: matrix.nrows(),
        n_cols: matrix.ncols(),
    };
    mann_whitney_sparse(smr, group1_indices, group2_indices, alternative)
}

/// Perform Mann-Whitney U tests on a sparse matrix represented by raw components.
pub fn mann_whitney_sparse<T, N, I>(
    matrix: SparseMatrixRef<T, N, I>,
    group1_indices: &[usize],
    group2_indices: &[usize],
    alternative: Alternative,
) -> anyhow::Result<Vec<TestResult<f64>>>
where
    T: FloatOpsTS,
    N: AsPrimitive<usize> + Send + Sync,
    I: AsPrimitive<usize> + Send + Sync,
    f64: std::convert::From<T>,
{
    if group1_indices.is_empty() || group2_indices.is_empty() {
        return Err(anyhow::anyhow!(
            "Single-Statistics | Group indices cannot be empty. Error code: SS-NP-001"
        ));
    }

    let nrows = matrix.n_rows;
    let n_group1 = group1_indices.len();
    let n_group2 = group2_indices.len();

    // Mapping from column index to group ID (0 for none, 1 for group1, 2 for group2)
    let mut cell_groups = vec![0u8; matrix.n_cols];
    for &idx in group1_indices {
        if idx < cell_groups.len() { cell_groups[idx] = 1; }
    }
    for &idx in group2_indices {
        if idx < cell_groups.len() { cell_groups[idx] = 2; }
    }

    let results: Vec<_> = (0..nrows)
        .into_par_iter()
        .map(|row| {
            let start = matrix.maj_ind[row].as_();
            let end = matrix.maj_ind[row + 1].as_();
            
            let mut x_nonzero = Vec::new();
            let mut y_nonzero = Vec::new();
            let mut g1_nz_count = 0;
            let mut g2_nz_count = 0;

            for i in start..end {
                let col = matrix.min_ind[i].as_();
                let val = f64::from(matrix.val[i]);
                
                match cell_groups[col] {
                    1 => {
                        if val != 0.0 { x_nonzero.push(val); }
                        g1_nz_count += 1;
                    },
                    2 => {
                        if val != 0.0 { y_nonzero.push(val); }
                        g2_nz_count += 1;
                    },
                    _ => {}
                }
            }
            
            let x_zeros = n_group1 - g1_nz_count;
            let y_zeros = n_group2 - g2_nz_count;

            mann_whitney_from_sparse_parts(x_nonzero, y_nonzero, x_zeros, y_zeros, alternative)
        })
        .collect();

    Ok(results)
}

/// Core MW-U logic optimized for sparse scRNA-seq data (many zeros).
fn mann_whitney_from_sparse_parts(
    x_nonzero: Vec<f64>,
    y_nonzero: Vec<f64>,
    x_zeros: usize,
    y_zeros: usize,
    alternative: Alternative,
) -> TestResult<f64> {
    let nx = x_zeros + x_nonzero.len();
    let ny = y_zeros + y_nonzero.len();

    if nx == 0 || ny == 0 {
        return TestResult::new(f64::NAN, 1.0);
    }

    let mut combined_nz: Vec<(f64, u8)> = Vec::with_capacity(x_nonzero.len() + y_nonzero.len());
    for v in x_nonzero { combined_nz.push((v, 0)); }
    for v in y_nonzero { combined_nz.push((v, 1)); }
    combined_nz.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    let n_total = (nx + ny) as f64;
    let (rank_sum_x, tie_correction) = {
        let n_zeros = (x_zeros + y_zeros) as f64;
        let mut rs_x = 0.0;
        let mut t_corr = 0.0;
        let mut current_rank = 1.0;

        // Handle Zeros
        if n_zeros > 0.0 {
            let avg_rank_zeros = (n_zeros + 1.0) / 2.0;
            rs_x += (x_zeros as f64) * avg_rank_zeros;
            t_corr += n_zeros.powi(3) - n_zeros;
            current_rank += n_zeros;
        }

        // Handle Non-zeros
        let mut i = 0;
        while i < combined_nz.len() {
            let val = combined_nz[i].0;
            let start = i;
            while i < combined_nz.len() && combined_nz[i].0 == val { i += 1; }
            let count = (i - start) as f64;
            let avg_rank = current_rank + (count - 1.0) / 2.0;
            
            for j in start..i {
                if combined_nz[j].1 == 0 { rs_x += avg_rank; }
            }
            if count > 1.0 {
                t_corr += count.powi(3) - count;
            }
            current_rank += count;
        }
        (rs_x, t_corr)
    };

    let nx_f = nx as f64;
    let ny_f = ny as f64;
    let u_x = rank_sum_x - (nx_f * (nx_f + 1.0)) / 2.0;
    let u_y = (nx_f * ny_f) - u_x;
    let mean_u = nx_f * ny_f / 2.0;
    
    let var_u = (nx_f * ny_f / (n_total * (n_total - 1.0))) * 
                ((n_total.powi(3) - n_total - tie_correction) / 12.0);

    let (u_stat, z) = match alternative {
        Alternative::TwoSided => {
            let u = u_x.min(u_y);
            let z_score = if var_u > 0.0 {
                ((u - mean_u).abs() - 0.5).max(0.0) / var_u.sqrt()
            } else { 0.0 };
            (u, z_score)
        },
        Alternative::Greater => {
            let z_score = if var_u > 0.0 {
                (u_x - mean_u - 0.5) / var_u.sqrt()
            } else { 0.0 };
            (u_x, z_score)
        },
        Alternative::Less => {
            let z_score = if var_u > 0.0 {
                (u_x - mean_u + 0.5) / var_u.sqrt()
            } else { 0.0 };
            (u_x, z_score)
        }
    };

    let p = calculate_p_value(z, alternative, nx_f, ny_f);
    TestResult::new(u_stat, p)
        .with_metadata("z_score", z)
        .with_metadata("var_u", var_u)
        .with_metadata("tie_correction", tie_correction)
}

/// Public API for two samples (dense).
pub fn mann_whitney_optimized(x: &[f64], y: &[f64], alternative: Alternative) -> TestResult<f64> {
    let mut x_nz = Vec::new();
    let mut x_z = 0;
    for &v in x { if v.is_finite() { if v == 0.0 { x_z += 1; } else { x_nz.push(v); } } }

    let mut y_nz = Vec::new();
    let mut y_z = 0;
    for &v in y { if v.is_finite() { if v == 0.0 { y_z += 1; } else { y_nz.push(v); } } }

    mann_whitney_from_sparse_parts(x_nz, y_nz, x_z, y_z, alternative)
}

#[inline]
fn calculate_p_value(z: f64, alternative: Alternative, nx: f64, ny: f64) -> f64 {
    if nx < 3.0 || ny < 3.0 { return 1.0; }
    if !z.is_finite() { return 1.0; }

    let normal = Normal::new(0.0, 1.0).unwrap();
    match alternative {
        Alternative::TwoSided => 2.0 * (1.0 - normal.cdf(z.abs())),
        Alternative::Greater => 1.0 - normal.cdf(z),
        Alternative::Less => normal.cdf(z),
    }
}
