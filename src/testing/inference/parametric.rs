use crate::testing::utils::accumulate_gene_statistics_two_groups;
use crate::testing::{Alternative, TTestType, TestResult};
use nalgebra_sparse::CsrMatrix;
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelIterator;
use single_utilities::traits::{FloatOps, FloatOpsTS};
use statrs::distribution::{ContinuousCDF, StudentsT};

pub fn t_test_matrix_groups<T>(
    matrix: &CsrMatrix<T>,
    group1_indices: &[usize],
    group2_indices: &[usize],
    test_type: TTestType,
) -> anyhow::Result<Vec<TestResult<T>>>
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
        accumulate_gene_statistics_two_groups(matrix, group1_indices, group2_indices)?;

    let results: Vec<TestResult<T>> = (0..n_genes)
        .into_iter()
        .map(|gene_idx| {
            fast_t_test_from_sums(
                group1_sums[gene_idx],
                group1_sum_squares[gene_idx],
                group1_size,
                group2_sums[gene_idx],
                group2_sum_squares[gene_idx],
                group2_size,
                test_type,
            )
        })
        .collect();

    Ok(results)
}

pub fn t_test<T>(x: &[T], y: &[T], test_type: TTestType, alternative: Alternative) -> TestResult<T>
where
    T: FloatOps,
{
    let nx = x.len();
    let ny = y.len();

    if nx < 2 || ny < 2 {
        return TestResult::new(T::zero(), T::one());
    }

    let (sum_x, sum_sq_x) = x
        .iter()
        .fold((T::zero(), T::zero()), |(sum, sum_sq), &val| {
            (sum + val, sum_sq + val * val)
        });

    let (sum_y, sum_sq_y) = y
        .iter()
        .fold((T::zero(), T::zero()), |(sum, sum_sq), &val| {
            (sum + val, sum_sq + val * val)
        });

    let nx_f = T::from(nx).unwrap();
    let ny_f = T::from(ny).unwrap();

    fast_t_test_from_sums(sum_x, sum_sq_x, nx_f, sum_y, sum_sq_y, ny_f, test_type)
}

fn fast_t_test_from_sums<T>(
    sum1: T,
    sum_sq1: T,
    n1: T,
    sum2: T,
    sum_sq2: T,
    n2: T,
    test_type: TTestType,
) -> TestResult<T>
where
    T: FloatOps,
{
    if n1 < T::from(2.0).unwrap() || n2 < T::from(2.0).unwrap() {
        return TestResult::new(T::zero(), T::one());
    }

    let mean1 = sum1 / n1;
    let mean2 = sum2 / n2;

    let var1 = (sum_sq1 - sum1 * mean1) / (n1 - T::one());
    let var2 = (sum_sq2 - sum2 * mean2) / (n2 - T::one());

    if var1 <= T::zero() && var2 <= T::zero() {
        if num_traits::Float::abs(mean1 - mean2) < <T as num_traits::Float>::epsilon() {
            return TestResult::new(T::zero(), T::one());
        } else {
            return TestResult::new(<T as num_traits::Float>::infinity(), T::zero());
        }
    }

    let (t_stat, df) = match test_type {
        TTestType::Student => {
            let pooled_var = ((n1 - T::one()) * var1 + (n2 - T::one()) * var2)
                / (n1 + n2 - T::from(2.0).unwrap());

            if pooled_var <= T::zero() {
                return TestResult::new(<T as num_traits::Float>::infinity(), T::zero());
            }

            let std_err = (pooled_var * (T::one() / n1 + T::one() / n2)).sqrt();
            let t = (mean1 - mean2) / std_err;
            (t, n1 + n2 - T::from(2.0).unwrap())
        }
        TTestType::Welch => {
            let term1 = var1 / n1;
            let term2 = var2 / n2;
            let combined_var = term1 + term2;

            if combined_var <= T::zero() {
                return TestResult::new(<T as num_traits::Float>::infinity(), T::zero());
            }

            let std_err = combined_var.sqrt();
            let t = (mean1 - mean2) / std_err;

            // Welch-Satterthwaite equation
            let df = combined_var * combined_var
                / (term1 * term1 / (n1 - T::one()) + term2 * term2 / (n2 - T::one()));
            (t, df)
        }
    };

    let p_value = fast_t_test_p_value(t_stat, df);

    TestResult::new(t_stat, p_value)
}

fn fast_t_test_p_value<T>(t_stat: T, df: T) -> T
where
    T: FloatOps,
{
    if !num_traits::Float::is_finite(t_stat) {
        return if num_traits::Float::is_infinite(t_stat) {
            T::zero()
        } else {
            T::one()
        };
    }

    if df <= T::zero() || !num_traits::Float::is_finite(df) {
        return T::one();
    }

    if df > T::from(30.0).unwrap() {
        let abs_t = num_traits::Float::abs(t_stat);
        return T::from(2.0).unwrap() * normal_cdf_complement(abs_t);
    }

    let t_f64 = t_stat.to_f64().unwrap();
    let df_f64 = df.to_f64().unwrap();

    match StudentsT::new(0.0, 1.0, df_f64) {
        Ok(t_dist) => {
            T::from(2.0).unwrap() * (T::one() - T::from(t_dist.cdf(t_f64.abs())).unwrap())
        }
        Err(_) => T::one(),
    }
}

fn calculate_p_value<T>(t_stat: T, df: T, alternative: Alternative) -> T
where
    T: FloatOps,
{
    let t_f64 = t_stat.to_f64().unwrap();
    let df_f64 = df.to_f64().unwrap();

    match StudentsT::new(0.0, 1.0, df_f64) {
        Ok(t_dist) => {
            let p = match alternative {
                Alternative::TwoSided => 2.0 * (1.0 - t_dist.cdf(t_f64.abs())),
                Alternative::Less => t_dist.cdf(t_f64),
                Alternative::Greater => 1.0 - t_dist.cdf(t_f64),
            };
            T::from(p).unwrap()
        }
        Err(_) => T::one(), // Fallback for invalid parameters
    }
}

pub fn student_t_quantile<T>(p: T, df: T) -> T
where
    T: FloatOps,
{
    if p <= T::zero() || p >= T::one() {
        panic!("Probability must be between 0 and 1 (exclusive)");
    }
    if df <= T::zero() {
        panic!("Degrees of freedom must be positive");
    }
    let df_f64 = df.to_f64().unwrap();
    let p_f64 = p.to_f64().unwrap();

    match StudentsT::new(0.0, 1.0, df_f64) {
        Ok(dist) => T::from(dist.inverse_cdf(p_f64)).unwrap(),
        Err(_) => panic!("Failed to create StudentsT distribution"),
    }
}

#[inline]
fn normal_cdf<T>(x: T) -> T
where
    T: FloatOps,
{
    if x < T::from(-8.0).unwrap() {
        return T::zero();
    }
    if x > T::from(8.0).unwrap() {
        return T::one();
    }

    T::from(0.5).unwrap() * (T::one() + erf_approx(x / T::from(2.0).unwrap().sqrt()))
}

#[inline]
fn normal_cdf_complement<T>(x: T) -> T
where
    T: FloatOps,
{
    if x < T::from(-8.0).unwrap() {
        return T::one();
    }
    if x > T::from(8.0).unwrap() {
        return T::zero();
    }

    T::from(0.5).unwrap() * erfc_approx(x / T::from(2.0).unwrap().sqrt())
}

#[inline]
fn erf_approx<T>(x: T) -> T
where
    T: FloatOps,
{
    let a1 = T::from(0.254829592).unwrap();
    let a2 = T::from(-0.284496736).unwrap();
    let a3 = T::from(1.421413741).unwrap();
    let a4 = T::from(-1.453152027).unwrap();
    let a5 = T::from(1.061405429).unwrap();
    let p = T::from(0.3275911).unwrap();

    let sign = if x < T::zero() {
        T::from(-1.0).unwrap()
    } else {
        T::one()
    };
    let x = num_traits::Float::abs(x);

    let t = T::one() / (T::one() + p * x);
    let y = T::one() - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[inline]
fn erfc_approx<T>(x: T) -> T
where
    T: FloatOps,
{
    T::one() - erf_approx(x)
}
