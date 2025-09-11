//! Parametric statistical tests for single-cell data analysis.
//!
//! This module implements parametric statistical tests, primarily t-tests, optimized for
//! sparse single-cell expression matrices. The implementations are designed for efficiency
//! when testing thousands of genes across different cell groups.

use crate::testing::utils::accumulate_gene_statistics_two_groups;
use crate::testing::{TTestType, TestResult};
use nalgebra_sparse::CsrMatrix;
use single_utilities::traits::{FloatOps, FloatOpsTS};
use statrs::distribution::{ContinuousCDF, StudentsT};

/// Perform t-tests on all genes comparing two groups of cells.
///
/// This is an optimized implementation that efficiently computes summary statistics
/// for sparse matrices and performs t-tests for each gene.
///
/// # Arguments
///
/// * `matrix` - Sparse expression matrix (genes × cells)
/// * `group1_indices` - Column indices for the first group of cells
/// * `group2_indices` - Column indices for the second group of cells
/// * `test_type` - Type of t-test to perform (Student's or Welch's)
///
/// # Returns
///
/// Vector of `TestResult` objects, one per gene, containing t-statistics and p-values.
pub fn t_test_matrix_groups<T>(
    matrix: &CsrMatrix<T>,
    group1_indices: &[usize],
    group2_indices: &[usize],
    test_type: TTestType,
) -> anyhow::Result<Vec<TestResult<f64>>>
where
    T: FloatOpsTS,
{
    if group1_indices.is_empty() || group2_indices.is_empty() {
        return Err(anyhow::anyhow!("Group indices cannot be empty"));
    }

    let n_genes = matrix.ncols();
    let group1_size = T::from(group1_indices.len()).unwrap();
    let group2_size = T::from(group2_indices.len()).unwrap();

        let (group1_sums, group1_sum_squares, group2_sums, group2_sum_squares) =
            accumulate_gene_statistics_two_groups(matrix, group1_indices, group2_indices)?;    let results: Vec<TestResult<f64>> = (0..n_genes)
        .map(|gene_idx| {
            fast_t_test_from_sums(
                group1_sums[gene_idx].to_f64().unwrap(),
                group1_sum_squares[gene_idx].to_f64().unwrap(),
                group1_size.to_f64().unwrap(),
                group2_sums[gene_idx].to_f64().unwrap(),
                group2_sum_squares[gene_idx].to_f64().unwrap(),
                group2_size.to_f64().unwrap(),
                test_type,
            )
        })
        .collect();

    Ok(results)
}

/// Perform a t-test comparing two samples.
///
/// This function performs either Student's t-test (assuming equal variances) or
/// Welch's t-test (allowing unequal variances) on two samples.
///
/// # Arguments
///
/// * `x` - First sample
/// * `y` - Second sample  
/// * `test_type` - Type of t-test to perform
///
/// # Returns
///
/// `TestResult` containing the t-statistic and p-value.
pub fn t_test<T>(x: &[T], y: &[T], test_type: TTestType) -> TestResult<f64>
where
    T: FloatOps,
{
    let nx = x.len();
    let ny = y.len();

    if nx < 2 || ny < 2 {
        return TestResult::new(0.0, 1.0);
    }

    // Branch optimization: use different strategies based on size
    if nx + ny < 1000 {
        // For small datasets, optimize for simplicity and cache locality
        t_test_small_optimized(x, y, test_type)
    } else {
        // For larger datasets, use the original approach
        t_test_large(x, y, test_type)
    }
}

#[inline]
fn t_test_small_optimized<T>(x: &[T], y: &[T], test_type: TTestType) -> TestResult<f64>
where
    T: FloatOps,
{
    // Optimized single-pass computation with better locality
    let mut sum_x = T::zero();
    let mut sum_sq_x = T::zero();
    for &val in x {
        sum_x += val;
        sum_sq_x += val * val;
    }

    let mut sum_y = T::zero();
    let mut sum_sq_y = T::zero();
    for &val in y {
        sum_y += val;
        sum_sq_y += val * val;
    }

    let nx_f = T::from(x.len()).unwrap();
    let ny_f = T::from(y.len()).unwrap();

    fast_t_test_from_sums(
        sum_x.to_f64().unwrap(), 
        sum_sq_x.to_f64().unwrap(), 
        nx_f.to_f64().unwrap(), 
        sum_y.to_f64().unwrap(), 
        sum_sq_y.to_f64().unwrap(), 
        ny_f.to_f64().unwrap(), 
        test_type
    )
}

#[inline]
fn t_test_large<T>(x: &[T], y: &[T], test_type: TTestType) -> TestResult<f64>
where
    T: FloatOps,
{
    // For larger datasets, use chunked processing to improve cache efficiency
    const CHUNK_SIZE: usize = 256;
    
    let mut sum_x = T::zero();
    let mut sum_sq_x = T::zero();
    
    for chunk in x.chunks(CHUNK_SIZE) {
        for &val in chunk {
            sum_x += val;
            sum_sq_x += val * val;
        }
    }

    let mut sum_y = T::zero();
    let mut sum_sq_y = T::zero();
    
    for chunk in y.chunks(CHUNK_SIZE) {
        for &val in chunk {
            sum_y += val;
            sum_sq_y += val * val;
        }
    }

    let nx_f = T::from(x.len()).unwrap();
    let ny_f = T::from(y.len()).unwrap();

    fast_t_test_from_sums(
        sum_x.to_f64().unwrap(), 
        sum_sq_x.to_f64().unwrap(), 
        nx_f.to_f64().unwrap(), 
        sum_y.to_f64().unwrap(), 
        sum_sq_y.to_f64().unwrap(), 
        ny_f.to_f64().unwrap(), 
        test_type
    )
}

/// Perform a t-test using precomputed summary statistics.
///
/// This is an optimized function that computes t-tests directly from sum and sum-of-squares,
/// avoiding the need to store or iterate through the original data. Particularly useful for
/// sparse matrix operations where computing these statistics is done efficiently during
/// matrix traversal.
///
/// # Arguments
///
/// * `sum1`, `sum_sq1`, `n1` - Sum, sum of squares, and count for group 1
/// * `sum2`, `sum_sq2`, `n2` - Sum, sum of squares, and count for group 2
/// * `test_type` - Type of t-test to perform (Student's or Welch's)
///
/// # Returns
///
/// `TestResult` containing the t-statistic and p-value.
pub fn fast_t_test_from_sums(
    sum1: f64,
    sum_sq1: f64,
    n1: f64,
    sum2: f64,
    sum_sq2: f64,
    n2: f64,
    test_type: TTestType,
) -> TestResult<f64>
{
    // Early exit for insufficient sample sizes
    if n1 < 2.0 || n2 < 2.0 {
        return TestResult::new(0.0, 1.0);
    }

    // Calculate means directly (avoiding redundant assignments)
    let mean1 = sum1 / n1;
    let mean2 = sum2 / n2;

    // Calculate variances using the computational formula
    let var1 = (sum_sq1 - sum1 * sum1 / n1) / (n1 - 1.0);
    let var2 = (sum_sq2 - sum2 * sum2 / n2) / (n2 - 1.0);
    
    let mean_diff = mean1 - mean2;
    
    let (t_stat, df) = match test_type {
        TTestType::Student => {
            // Student's t-test (pooled variance)
            let pooled_var = ((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / (n1 + n2 - 2.0);
            let std_err = (pooled_var * (1.0 / n1 + 1.0 / n2)).sqrt();
            (mean_diff / std_err, n1 + n2 - 2.0)
        }
        TTestType::Welch => {
            // Welch's t-test (unequal variances)
            let term1 = var1 / n1;
            let term2 = var2 / n2;
            let combined_var = term1 + term2;
            let std_err = combined_var.sqrt();
            let t = mean_diff / std_err;
            
            // Welch-Satterthwaite equation for degrees of freedom
            let df = combined_var * combined_var / 
                (term1 * term1 / (n1 - 1.0) + term2 * term2 / (n2 - 1.0));
            (t, df)
        }
    };

    let p_value = fast_t_test_p_value(t_stat, df);
    TestResult::new(t_stat, p_value)
}

#[inline]
fn fast_t_test_p_value(t_stat: f64, df: f64) -> f64
{
    // Fast path for non-finite inputs
    if !t_stat.is_finite() {
        return if t_stat.is_infinite() { 0.0 } else { 1.0 };
    }

    if df <= 0.0 || !df.is_finite() {
        return 1.0;
    }

    let abs_t = t_stat.abs();

    // Fast path for very small t-statistics (common case)
    if abs_t < 0.001 {
        return 1.0; // p-value ≈ 1 for very small effects
    }

    // Early return for very large t-statistics (avoids expensive computations)
    if abs_t > 37.0 {
        let log_p = log_normal_tail_probability(abs_t);
        return 2.0 * log_p.exp();
    }

    // Use normal approximation for large degrees of freedom (faster than t-distribution)
    if df > 100.0 {
        return 2.0 * high_precision_normal_cdf_complement(abs_t);
    }

    // Only create StudentsT distribution when necessary
    match StudentsT::new(0.0, 1.0, df) {
        Ok(t_dist) => {
            let cdf_val = t_dist.cdf(abs_t);
            2.0 * (1.0 - cdf_val)
        }
        Err(_) => 1.0,
    }
}

/// High-precision calculation of log(P(Z > x)) for standard normal
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

/// High-precision complementary error function for extreme values
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

/// Continued fraction approximation for log(erfc(x))
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

/// Series expansion for erf(x) for small x
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

/// High-precision normal CDF complement for extreme values
#[inline]
fn high_precision_normal_cdf_complement(x: f64) -> f64 {
    if x < 0.0 {
        return 1.0 - high_precision_normal_cdf_complement(-x);
    }
    
    if x > 37.0 {
        let log_p = log_normal_tail_probability(x);
        return log_p.exp();
    }
    
    0.5 * erfc_high_precision(x / (2.0_f64).sqrt())
}

/// High-precision complementary error function
#[inline]
fn erfc_high_precision(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc_high_precision(-x);
    }
    
    if x > 26.0 {
        return 0.0; 
    }
    
    if x < 2.0 {
        return 1.0 - erf_series(x);
    }
    chebyshev_erfc(x)
}

/// Chebyshev rational approximation for erfc
#[inline]
fn chebyshev_erfc(x: f64) -> f64 {
    let a1 = 0.0705230784;
    let a2 = 0.0422820123;
    let a3 = 0.0092705272;
    let a4 = 0.0001520143;
    let a5 = 0.0002765672;
    let a6 = 0.0000430638;
    
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * (a5 + t * a6)))));
    
    poly * (-x * x).exp()
}

