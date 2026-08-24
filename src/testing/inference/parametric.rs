//! Parametric statistical tests for single-cell data analysis.
//!
//! This module implements parametric statistical tests, primarily t-tests, optimized for
//! sparse single-cell expression matrices. The implementations are designed for efficiency
//! when testing thousands of genes across different cell groups.

use crate::testing::utils::{
    accumulate_gene_statistics_two_groups_raw, standard_normal, SparseMatrixRef, SprsView,
};
use crate::testing::{TTestType, TestResult};
use single_utilities::traits::{FloatOps, FloatOpsTS};
use statrs::distribution::{ContinuousCDF, StudentsT};
use num_traits::AsPrimitive;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Perform t-tests on all genes comparing two groups of cells.
///
/// `matrix` is genes (major axis) × cells (minor axis); the group slices index cells.
/// Returns one `TestResult` per gene.
pub fn t_test_matrix_groups<T, I, Iptr, IptrStorage, IndStorage, DataStorage>(
    matrix: &sprs::CsMatBase<T, I, IptrStorage, IndStorage, DataStorage, Iptr>,
    group1_indices: &[usize],
    group2_indices: &[usize],
    test_type: TTestType,
) -> anyhow::Result<Vec<TestResult<f64>>>
where
    T: FloatOpsTS,
    I: sprs::indexing::SpIndex + AsPrimitive<usize>,
    Iptr: sprs::indexing::SpIndex + AsPrimitive<usize>,
    IptrStorage: std::ops::Deref<Target = [Iptr]> + Send + Sync,
    IndStorage: std::ops::Deref<Target = [I]> + Send + Sync,
    DataStorage: std::ops::Deref<Target = [T]> + Send + Sync,
{
    t_test_sparse(
        SprsView::new(matrix).as_matrix_ref(),
        group1_indices,
        group2_indices,
        test_type,
    )
}

/// Perform t-tests on a sparse matrix represented by raw components.
/// 
/// This version is agnostic of the matrix container and can be used with raw vectors.
pub fn t_test_sparse<T, N, I>(
    matrix: SparseMatrixRef<T, N, I>,
    group1_indices: &[usize],
    group2_indices: &[usize],
    test_type: TTestType,
) -> anyhow::Result<Vec<TestResult<f64>>>
where
    T: FloatOpsTS,
    N: AsPrimitive<usize> + Send + Sync,
    I: AsPrimitive<usize> + Send + Sync,
{
    if group1_indices.is_empty() || group2_indices.is_empty() {
        return Err(anyhow::anyhow!("Group indices cannot be empty"));
    }

    let n_genes = matrix.n_rows;
    let group1_size = T::from(group1_indices.len()).unwrap();
    let group2_size = T::from(group2_indices.len()).unwrap();

    let (group1_sums, group1_sum_squares, group2_sums, group2_sum_squares) =
        accumulate_gene_statistics_two_groups_raw(matrix, group1_indices, group2_indices)?;

    let n1 = group1_size.to_f64().unwrap();
    let n2 = group2_size.to_f64().unwrap();

    let results: Vec<TestResult<f64>> = (0..n_genes)
        .into_par_iter()
        .map(|gene_idx| {
            fast_t_test_from_sums(
                group1_sums[gene_idx].to_f64().unwrap(),
                group1_sum_squares[gene_idx].to_f64().unwrap(),
                n1,
                group2_sums[gene_idx].to_f64().unwrap(),
                group2_sum_squares[gene_idx].to_f64().unwrap(),
                n2,
                test_type,
            )
        })
        .collect();

    Ok(results)
}

/// Perform a t-test comparing two samples: Student's (equal variances) or Welch's.
pub fn t_test<T>(x: &[T], y: &[T], test_type: TTestType) -> TestResult<f64>
where
    T: FloatOps,
{
    if x.len() < 2 || y.len() < 2 {
        return TestResult::new(0.0, 1.0);
    }

    let (sum_x, sum_sq_x) = sum_and_sum_of_squares(x);
    let (sum_y, sum_sq_y) = sum_and_sum_of_squares(y);

    fast_t_test_from_sums(
        sum_x,
        sum_sq_x,
        x.len() as f64,
        sum_y,
        sum_sq_y,
        y.len() as f64,
        test_type,
    )
}

/// Single-pass accumulation of the sum and sum of squares of a sample.
#[inline]
fn sum_and_sum_of_squares<T>(v: &[T]) -> (f64, f64)
where
    T: FloatOps,
{
    let mut sum = T::zero();
    let mut sum_sq = T::zero();
    for &val in v {
        sum += val;
        sum_sq += val * val;
    }
    (sum.to_f64().unwrap(), sum_sq.to_f64().unwrap())
}

/// t-test from precomputed sums, so the matrix never has to be revisited.
///
/// Takes (sum, sum of squares, count) per group.
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
    if n1 < 2.0 || n2 < 2.0 {
        return TestResult::new(0.0, 1.0);
    }

    let mean1 = sum1 / n1;
    let mean2 = sum2 / n2;

    // Variance via the computational formula. It cancels badly when the mean is
    // large relative to the spread, which can drive the numerator slightly negative;
    // clamping at zero keeps a marginal case from turning into a NaN t-statistic.
    let var1 = ((sum_sq1 - sum1 * sum1 / n1) / (n1 - 1.0)).max(0.0);
    let var2 = ((sum_sq2 - sum2 * sum2 / n2) / (n2 - 1.0)).max(0.0);
    
    let mean_diff = mean1 - mean2;
    
    let (std_err, df) = match test_type {
        TTestType::Student => {
            let pooled_var = ((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / (n1 + n2 - 2.0);
            ((pooled_var * (1.0 / n1 + 1.0 / n2)).sqrt(), n1 + n2 - 2.0)
        }
        TTestType::Welch => {
            let term1 = var1 / n1;
            let term2 = var2 / n2;
            let combined_var = term1 + term2;

            // Welch-Satterthwaite equation for degrees of freedom
            let df = combined_var * combined_var
                / (term1 * term1 / (n1 - 1.0) + term2 * term2 / (n2 - 1.0));
            (combined_var.sqrt(), df)
        }
    };

    let t_stat = mean_diff / std_err;
    let p_value = fast_t_test_p_value(t_stat, df);
    let (lo, hi) = mean_difference_ci(mean_diff, std_err, df);

    TestResult::new(t_stat, p_value)
        .with_degrees_of_freedom(df)
        .with_standard_error(std_err)
        .with_confidence_interval(lo, hi)
}

/// 95% confidence interval on the difference in means.
///
/// Uses the normal quantile above 100 df, matching what the p-value path already
/// does there and keeping the per-gene cost to a constant.
#[inline]
fn mean_difference_ci(mean_diff: f64, std_err: f64, df: f64) -> (f64, f64) {
    if !std_err.is_finite() || std_err <= 0.0 || df <= 0.0 || !df.is_finite() {
        return (f64::NAN, f64::NAN);
    }

    const Z_975: f64 = 1.959_963_984_540_054;
    let crit = if df > 100.0 {
        Z_975
    } else {
        match StudentsT::new(0.0, 1.0, df) {
            Ok(d) => d.inverse_cdf(0.975),
            Err(_) => return (f64::NAN, f64::NAN),
        }
    };

    (mean_diff - crit * std_err, mean_diff + crit * std_err)
}

#[inline]
fn fast_t_test_p_value(t_stat: f64, df: f64) -> f64 {
    if !t_stat.is_finite() {
        return if t_stat.is_infinite() { 0.0 } else { 1.0 };
    }
    if df <= 0.0 || !df.is_finite() {
        return 1.0;
    }

    let abs_t = t_stat.abs();
    if abs_t < 0.001 {
        return 1.0;
    }

    // `sf` is the upper tail computed directly, so it stays accurate where
    // `1 - cdf` would cancel away to zero.
    if df > 100.0 {
        return (2.0 * standard_normal().sf(abs_t)).clamp(0.0, 1.0);
    }

    match StudentsT::new(0.0, 1.0, df) {
        Ok(t_dist) => (2.0 * t_dist.sf(abs_t)).clamp(0.0, 1.0),
        Err(_) => 1.0,
    }
}
