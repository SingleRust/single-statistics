//! Discrete statistical tests for single-cell data analysis.
//! 
//! This module implements tests for categorical and count data,
//! such as Chi-square and Fisher's exact tests.

use crate::testing::{Alternative, TestResult};
use single_utilities::traits::FloatOpsTS;
use statrs::distribution::{ChiSquared, ContinuousCDF, Discrete, DiscreteCDF};
use crate::testing::utils::SparseMatrixRef;
use num_traits::{AsPrimitive, Float};
use rayon::prelude::*;

/// Performs a chi-square test for independence on a 2x2 contingency table
pub fn chi_square_test<T>(
    a: T,
    b: T,
    c: T,
    d: T,
    alternative: Alternative,
) -> TestResult<T>
where
    T: FloatOpsTS,
{
    let total = a + b + c + d;
    if total <= T::zero() {
        return TestResult::new(T::zero(), T::one());
    }

    // Calculate expected frequencies
    let row1 = a + b;
    let row2 = c + d;
    let col1 = a + c;
    let col2 = b + d;

    let expected_a = (row1 * col1) / total;
    let expected_b = (row1 * col2) / total;
    let expected_c = (row2 * col1) / total;
    let expected_d = (row2 * col2) / total;

    // Calculate chi-square statistic
    let chi_square = (Float::powi(a - expected_a, 2) / expected_a)
        + (Float::powi(b - expected_b, 2) / expected_b)
        + (Float::powi(c - expected_c, 2) / expected_c)
        + (Float::powi(d - expected_d, 2) / expected_d);

    // Calculate p-value using chi-square distribution with 1 degree of freedom
    let p_value = calculate_chi_square_p_value(chi_square, T::one(), alternative);

    TestResult::new(chi_square, p_value)
}

fn calculate_chi_square_p_value<T>(chi_square: T, df: T, alternative: Alternative) -> T
where
    T: FloatOpsTS,
{
    let chi_square_f64 = chi_square.to_f64().unwrap();
    let df_f64 = df.to_f64().unwrap();

    match ChiSquared::new(df_f64) {
        Ok(chi_dist) => {
            let p = match alternative {
                Alternative::TwoSided => 1.0 - chi_dist.cdf(chi_square_f64), // Chi-square is usually 1-tailed
                Alternative::Less => chi_dist.cdf(chi_square_f64),
                Alternative::Greater => 1.0 - chi_dist.cdf(chi_square_f64),
            };
            T::from(p).unwrap()
        }
        Err(_) => T::one(),
    }
}

/// Fisher's Exact Test for 2x2 contingency table.
/// 
/// Hypergeometric distribution: 
/// N: total balls, K: total white balls, n: balls drawn, k: white balls drawn
/// 
/// Contingency table:
///         Group1  Group2
/// Expr      a       b
/// NonExpr   c       d
pub fn fisher_exact_test<T>(
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    _alternative: Alternative,
) -> TestResult<T>
where
    T: FloatOpsTS,
{
    // Implementation uses statrs Hypergeometric distribution
    use statrs::distribution::Hypergeometric;
    
    let n1 = a + c; // Group 1 size
    let n2 = b + d; // Group 2 size
    let total_expr = a + b;
    let total_cells = n1 + n2;

    if total_cells == 0 {
        return TestResult::new(T::zero(), T::one());
    }

    // Hypergeometric(total, success_in_total, draws)
    // Here: N=total_cells, K=total_expr, n=n1 (draws from group 1)
    match Hypergeometric::new(total_cells as u64, total_expr as u64, n1 as u64) {
        Ok(hyper) => {
            let p_val: f64 = match _alternative {
                Alternative::Greater => 1.0 - hyper.cdf((a as u64).saturating_sub(1)),
                Alternative::Less => hyper.cdf(a as u64),
                Alternative::TwoSided => {
                    let p_a = hyper.pmf(a as u64);
                    let mut p_sum = 0.0;
                    let upper_limit = std::cmp::min(n1, total_expr);
                    for i in 0..=upper_limit {
                        let p_i = hyper.pmf(i as u64);
                        if p_i <= p_a + 1e-12 {
                            p_sum += p_i;
                        }
                    }
                    p_sum.min(1.0)
                }
            };
            
            let odds_ratio = if b * c == 0 {
                if a * d > 0 { f64::INFINITY } else { 0.0 }
            } else {
                (a as f64 * d as f64) / (b as f64 * c as f64)
            };
            
            TestResult::new(T::from(odds_ratio).unwrap(), T::from(p_val).unwrap())
        }
        Err(_) => TestResult::new(T::zero(), T::one()),
    }
}

/// Perform Fisher's Exact Test across all genes in a sparse matrix.
pub fn fisher_exact_sparse<T, N, I>(
    matrix: SparseMatrixRef<T, N, I>,
    group1_indices: &[usize],
    group2_indices: &[usize],
    alternative: Alternative,
) -> anyhow::Result<Vec<TestResult<T>>>
where
    T: FloatOpsTS,
    N: AsPrimitive<usize> + Send + Sync,
    I: AsPrimitive<usize> + Send + Sync,
{
    let n_group1 = group1_indices.len();
    let n_group2 = group2_indices.len();
    
    let mut cell_groups = vec![0u8; matrix.n_cols];
    for &idx in group1_indices { if idx < cell_groups.len() { cell_groups[idx] = 1; } }
    for &idx in group2_indices { if idx < cell_groups.len() { cell_groups[idx] = 2; } }

    let results: Vec<_> = (0..matrix.n_rows)
        .into_par_iter()
        .map(|row| {
            let start = matrix.maj_ind[row].as_();
            let end = matrix.maj_ind[row + 1].as_();
            
            let mut a = 0; // Group 1 Expressed
            let mut b = 0; // Group 2 Expressed

            for i in start..end {
                let col = matrix.min_ind[i].as_();
                match cell_groups[col] {
                    1 => a += 1,
                    2 => b += 1,
                    _ => {}
                }
            }
            
            let c = n_group1 - a;
            let d = n_group2 - b;

            fisher_exact_test(a, b, c, d, alternative)
        })
        .collect();

    Ok(results)
}
