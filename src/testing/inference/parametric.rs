use crate::testing::{Alternative, TTestType, TestResult};
use nalgebra_sparse::CsrMatrix;
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
    let mut results = Vec::with_capacity(n_genes);

    // Pre-allocate working vectors
    let mut group1_values = Vec::with_capacity(group1_indices.len());
    let mut group2_values = Vec::with_capacity(group2_indices.len());

    for gene_idx in 0..n_genes {
        // Clear and reuse vectors
        group1_values.clear();
        group2_values.clear();

        // Extract values for this gene (column gene_idx)
        // For each cell in group1, get the gene expression value
        for &cell_idx in group1_indices {
            let value = if let Some(entry) = matrix.get_entry(cell_idx, gene_idx) {
                entry.into_value()
            } else {
                T::zero() // Handle sparse entries
            };
            group1_values.push(value);
        }

        // For each cell in group2, get the gene expression value
        for &cell_idx in group2_indices {
            let value = if let Some(entry) = matrix.get_entry(cell_idx, gene_idx) {
                entry.into_value()
            } else {
                T::zero() // Handle sparse entries
            };
            group2_values.push(value);
        }

        // Run t-test for this gene
        let result = t_test(
            &group1_values,
            &group2_values,
            test_type,
            Alternative::TwoSided,
        );
        results.push(result);
    }

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

    // Calculate means
    let sum_x: T = x.iter().copied().sum();
    let sum_y: T = y.iter().copied().sum();

    let nx_f = T::from(nx).unwrap();
    let ny_f = T::from(ny).unwrap();

    let mean_x = sum_x / nx_f;
    let mean_y = sum_y / ny_f;

    // Calculate sample variances using the correct formula
    let var_x = x.iter()
        .map(|&val| (val - mean_x) * (val - mean_x))
        .sum::<T>() / (nx_f - T::one());

    let var_y = y.iter()
        .map(|&val| (val - mean_y) * (val - mean_y))
        .sum::<T>() / (ny_f - T::one());

    // Early exit for zero variance cases
    if var_x <= T::zero() && var_y <= T::zero() {
        if num_traits::Float::abs(mean_x - mean_y) < <T as num_traits::Float>::epsilon() {
            return TestResult::new(T::zero(), T::one()); // No difference, no variance
        } else {
            return TestResult::new(<T as num_traits::Float>::infinity(), T::zero()); // Infinite t-stat, highly significant
        }
    }

    let (t_stat, df) = match test_type {
        TTestType::Student => {
            // Pooled variance (equal variances assumed)
            let pooled_var = ((nx_f - T::one()) * var_x + (ny_f - T::one()) * var_y)
                / (nx_f + ny_f - T::from(2.0).unwrap());

            if pooled_var <= T::zero() {
                return TestResult::new(<T as num_traits::Float>::infinity(), T::zero());
            }

            let std_err = (pooled_var * (T::one() / nx_f + T::one() / ny_f)).sqrt();
            let t = (mean_x - mean_y) / std_err;
            let degrees_freedom = nx_f + ny_f - T::from(2.0).unwrap();
            (t, degrees_freedom)
        }
        TTestType::Welch => {
            // Welch's t-test (unequal variances)
            let term1 = var_x / nx_f;
            let term2 = var_y / ny_f;
            let combined_var = term1 + term2;

            if combined_var <= T::zero() {
                return TestResult::new(<T as num_traits::Float>::infinity(), T::zero());
            }

            let std_err = combined_var.sqrt();
            let t = (mean_x - mean_y) / std_err;

            // Welch-Satterthwaite equation for degrees of freedom
            let df = combined_var * combined_var
                / (term1 * term1 / (nx_f - T::one()) + term2 * term2 / (ny_f - T::one()));
            (t, df)
        }
    };

    // Handle edge cases
    if !num_traits::Float::is_finite(t_stat) {
        return TestResult::new(
            t_stat,
            if num_traits::Float::is_infinite(t_stat) {
                T::zero()
            } else {
                T::one()
            },
        );
    }

    if df <= T::zero() || !num_traits::Float::is_finite(df) {
        return TestResult::new(t_stat, T::one());
    }

    // Calculate p-value using t-distribution
    let p_value = calculate_p_value(t_stat, df, alternative);

    TestResult::new(t_stat, num_traits::Float::clamp(p_value, T::zero(), T::one()))
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
                Alternative::TwoSided => {
                    // Two-tailed test
                    2.0 * (1.0 - t_dist.cdf(t_f64.abs()))
                }
                Alternative::Less => {
                    // Left-tailed test: P(T <= t)
                    t_dist.cdf(t_f64)
                }
                Alternative::Greater => {
                    // Right-tailed test: P(T >= t)
                    1.0 - t_dist.cdf(t_f64)
                }
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

    // Create a Student's t distribution with the specified degrees of freedom
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
