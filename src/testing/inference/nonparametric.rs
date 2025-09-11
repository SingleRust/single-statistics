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

#[derive(Debug, Clone)]
struct TieInfo {
    tie_counts: Vec<usize>,
    tie_correction: f64,
}

/// Perform Mann-Whitney U tests on all genes comparing two groups of cells.
///
/// This function efficiently computes Mann-Whitney U statistics for all genes in a sparse
/// matrix, comparing expression distributions between two groups of cells. The implementation
/// uses parallel processing for improved performance on large datasets.
///
/// # Arguments
///
/// * `matrix` - Sparse expression matrix (genes × cells)
/// * `group1_indices` - Column indices for the first group of cells
/// * `group2_indices` - Column indices for the second group of cells
/// * `alternative` - Type of alternative hypothesis (two-sided, less, greater)
///
/// # Returns
///
/// Vector of `TestResult` objects containing U statistics and p-values for each gene.
/// let group1 = vec![0, 1, 2]; // First group of cells
/// let group2 = vec![3, 4, 5]; // Second group of cells
///
/// let results = mann_whitney_matrix_groups(
///     &matrix, 
///     &group1, 
///     &group2, 
///     Alternative::TwoSided
/// )?;
/// # Ok(())
/// # }
/// ```
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
    if group1_indices.is_empty() || group2_indices.is_empty() {
        return Err(anyhow::anyhow!(
            "Single-Statistics | Group indices cannot be empty. Error code: SS-NP-001"
        ));
    }

    // Simplified: use the same implementation regardless of size
    let nrows = matrix.nrows();

    let results: Vec<_> = (0..nrows)
        .into_par_iter()
        .map(|row| {
            let mut group1_values: Vec<f64> = Vec::with_capacity(group1_indices.len());
            let mut group2_values: Vec<f64> = Vec::with_capacity(group2_indices.len());

            extract_row_values(matrix, row, group1_indices, &mut group1_values);
            extract_row_values(matrix, row, group2_indices, &mut group2_values);

            mann_whitney_optimized(&group1_values, &group2_values, alternative)
        })
        .collect();

    Ok(results)
}

/// Perform an optimized Mann-Whitney U test on two samples.
///
/// This function computes the Mann-Whitney U statistic and p-value for comparing two
/// independent samples. It handles ties correctly and supports different alternative
/// hypotheses.
///
/// # Arguments
///
/// * `x` - First sample
/// * `y` - Second sample
/// * `alternative` - Type of alternative hypothesis
///
/// # Returns
///
/// `TestResult` containing the U statistic and p-value.
///
/// # Example
///
/// ```rust
/// use single_statistics::testing::inference::nonparametric::mann_whitney_optimized;
/// use single_statistics::testing::Alternative;
///
/// let group1 = vec![1.0, 2.0, 3.0];
/// let group2 = vec![4.0, 5.0, 6.0];
/// let result = mann_whitney_optimized(&group1, &group2, Alternative::TwoSided);
/// 
/// println!("U statistic: {}, p-value: {}", result.statistic, result.p_value);
/// ```
pub fn mann_whitney_optimized(x: &[f64], y: &[f64], alternative: Alternative) -> TestResult<f64> {
    let nx = x.len();
    let ny = y.len();

    if nx == 0 || ny == 0 {
        return TestResult::new(f64::NAN, 1.0);
    }

    if nx == 1 && ny == 1 {
        let (u, p_value) = if x[0] > y[0] {
            (
                1.0,
                match alternative {
                    Alternative::Greater => 0.5,
                    Alternative::Less => 1.0,
                    Alternative::TwoSided => 1.0,
                },
            )
        } else if x[0] < y[0] {
            (
                0.0,
                match alternative {
                    Alternative::Greater => 1.0,
                    Alternative::Less => 0.5,
                    Alternative::TwoSided => 1.0,
                },
            )
        } else {
            (0.5, 1.0)
        };
        return TestResult::new(u, p_value);
    }

    let total_size = nx + ny;
    let mut combined: Vec<(f64, u8)> = Vec::with_capacity(total_size);

    let mut valid_nx = 0;
    let mut valid_ny = 0;

    for &v in x {
        if v.is_finite() {
            combined.push((v, 0));
            valid_nx += 1;
        }
    }

    for &v in y {
        if v.is_finite() {
            combined.push((v, 1));
            valid_ny += 1;
        }
    }

    if valid_nx == 0 || valid_ny == 0 {
        return TestResult::new(f64::NAN, 1.0);
    }

    combined.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    let (rank_sum_x, tie_info) = calculate_rank_sum_with_ties(&combined, valid_nx, valid_ny);

    let nx_f64 = valid_nx as f64;
    let ny_f64 = valid_ny as f64;

    let u_x = rank_sum_x - (nx_f64 * (nx_f64 + 1.0)) / 2.0;
    let u_y = (nx_f64 * ny_f64) - u_x;

    let mean_u = nx_f64 * ny_f64 / 2.0;

    let n_total = nx_f64 + ny_f64;
    let var_u = if tie_info.tie_correction > 0.0 {
        (nx_f64 * ny_f64 / 12.0)
            * (n_total + 1.0 - tie_info.tie_correction / (n_total * (n_total - 1.0)))
    } else {
        nx_f64 * ny_f64 * (n_total + 1.0) / 12.0
    };

    let (u_statistic, z_score) = match alternative {
        Alternative::TwoSided => {
            let u = u_x.min(u_y);
            let z = if var_u > 0.0 {
                let corrected_u = if u < mean_u { u + 0.5 } else { u - 0.5 };
                (mean_u - corrected_u) / var_u.sqrt()
            } else {
                0.0
            };
            (u, z.abs())
        }
        Alternative::Less => {
            let z = if var_u > 0.0 {
                (mean_u - u_x - 0.5) / var_u.sqrt()
            } else {
                0.0
            };
            (u_x, z)
        }
        Alternative::Greater => {
            let z = if var_u > 0.0 {
                (u_x - mean_u - 0.5) / var_u.sqrt()
            } else {
                0.0
            };
            (u_x, z)
        }
    };

    let p_value = calculate_p_value(z_score, alternative, nx_f64, ny_f64);

    let effect_size = if nx_f64 + ny_f64 > 0.0 {
        z_score / (nx_f64 + ny_f64).sqrt()
    } else {
        0.0
    };

    let standard_error = var_u.sqrt();

    TestResult::with_effect_size(u_statistic, p_value, effect_size)
        .with_standard_error(standard_error)
        .with_metadata("z_score", z_score.abs())
        .with_metadata("mean_u", mean_u)
        .with_metadata("var_u", var_u)
        .with_metadata("nx", nx_f64)
        .with_metadata("ny", ny_f64)
        .with_metadata("tie_correction", tie_info.tie_correction)
}

#[inline]
fn calculate_p_value(z: f64, alternative: Alternative, nx: f64, ny: f64) -> f64 {
    if nx < 3.0 || ny < 3.0 {
        return 1.0;
    }

    if !z.is_finite() {
        return 1.0;
    }

    if z.abs() > 37.0 {
        let log_p = log_normal_tail_probability(z.abs());
        return match alternative {
            Alternative::TwoSided => (2.0 * log_p.exp()).min(1.0),
            Alternative::Less | Alternative::Greater => log_p.exp().min(1.0),
        };
    }

    match Normal::new(0.0, 1.0) {
        Ok(normal) => match alternative {
            Alternative::TwoSided => 2.0 * (1.0 - normal.cdf(z.abs())).min(0.5),
            Alternative::Less => normal.cdf(z),
            Alternative::Greater => 1.0 - normal.cdf(z),
        },
        Err(_) => 1.0,
    }
}

#[inline]
fn calculate_rank_sum_with_ties(combined: &[(f64, u8)], nx: usize, ny: usize) -> (f64, TieInfo) {
    let mut rank_sum_x = 0.0;
    let mut tie_counts = Vec::new();
    let mut tie_correction = 0.0;
    let mut i = 0;
    let len = combined.len();

    while i < len {
        let val = combined[i].0;
        let tie_start = i;

        // Count ties
        while i < len && combined[i].0 == val {
            i += 1;
        }

        let tie_end = i;
        let tie_size = tie_end - tie_start;

        // Average rank for tied values
        let avg_rank = (tie_start + tie_end - 1) as f64 / 2.0 + 1.0;

        // Accumulate rank sum for group x
        for j in tie_start..tie_end {
            if combined[j].1 == 0 {
                rank_sum_x += avg_rank;
            }
        }

        // Calculate tie correction if there are ties
        if tie_size > 1 {
            tie_counts.push(tie_size);
            let t = tie_size as f64;
            tie_correction += t * (t * t - 1.0);
        }
    }

    (
        rank_sum_x,
        TieInfo {
            tie_counts,
            tie_correction,
        },
    )
}

#[inline]
fn log_normal_tail_probability(x: f64) -> f64 {
    if x < 0.0 {
        return 0.0;
    }

    if x > 8.0 {
        let x_sq = x * x;
        return -0.5 * x_sq - (x * (2.0 * std::f64::consts::PI).sqrt()).ln();
    }

    let z = x / (2.0_f64).sqrt();
    log_erfc(z) - (2.0_f64).ln()
}

#[inline]
fn log_erfc(x: f64) -> f64 {
    if x < 0.0 {
        return 0.0;
    }

    if x > 26.0 {
        let x_sq = x * x;
        return -x_sq - 0.5 * (std::f64::consts::PI).ln() - x.ln();
    }

    continued_fraction_log_erfc(x)
}

#[inline]
fn continued_fraction_log_erfc(x: f64) -> f64 {
    if x < 2.0 {
        let erf_val = erf_series(x);
        return (1.0 - erf_val).ln();
    }

    let x_sq = x * x;
    let mut a = 1.0;
    let mut b = 2.0 * x_sq;
    let mut result = a / b;

    for n in 1..50 {
        a = -(2 * n - 1) as f64;
        b = 2.0 * x_sq + a / result;
        let new_result = a / b;

        if (result - new_result).abs() < 1e-15 {
            break;
        }
        result = new_result;
    }

    -x_sq + (result / (x * (std::f64::consts::PI).sqrt())).ln()
}

#[inline]
fn erf_series(x: f64) -> f64 {
    let x_sq = x * x;
    let mut term = x;
    let mut result = term;

    for n in 1..100 {
        term *= -x_sq / (n as f64);
        let new_term = term / (2.0 * n as f64 + 1.0);
        result += new_term;

        if new_term.abs() < 1e-16 {
            break;
        }
    }

    result * 2.0 / (std::f64::consts::PI).sqrt()
}

#[inline]
fn extract_row_values<T>(
    matrix: &CsrMatrix<T>,
    row: usize,
    indices: &[usize],
    values: &mut Vec<f64>,
) where
    T: FloatOpsTS,
    f64: std::convert::From<T>,
{
    // Get row slice for efficient access
    let row_view = matrix.row(row);

    for &col in indices {
        if let Some(value) = row_view.get_entry(col) {
            values.push(value.into_value().into());
        } else {
            values.push(0.0);
        }
    }
}
