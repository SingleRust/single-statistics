//! Discrete statistical tests for single-cell data analysis.
//! 
//! This module implements tests for categorical and count data,
//! such as Chi-square and Fisher's exact tests.

use crate::testing::{Alternative, TTestType, TestResult};
use single_utilities::traits::FloatOpsTS;
use statrs::distribution::{ChiSquared, ContinuousCDF, Discrete, DiscreteCDF, Normal};
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

    // A zero row or column total makes the corresponding expected counts zero.
    // Dividing by them would yield inf/NaN, so those cells contribute nothing.
    let term = |obs: T, exp: T| {
        if exp > T::zero() {
            Float::powi(obs - exp, 2) / exp
        } else {
            T::zero()
        }
    };

    // Calculate chi-square statistic
    let chi_square = term(a, expected_a)
        + term(b, expected_b)
        + term(c, expected_c)
        + term(d, expected_d);

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

/// Chi-square goodness-of-fit test comparing observed counts against expected ones.
///
/// Categories with a non-positive expected count are skipped rather than producing
/// a division by zero. Degrees of freedom are `len - 1`.
pub fn chi_square_goodness_of_fit<T>(
    observed: &[T],
    expected: &[T],
    alternative: Alternative,
) -> TestResult<T>
where
    T: FloatOpsTS,
{
    if observed.len() != expected.len() || observed.len() < 2 {
        return TestResult::new(T::zero(), T::one());
    }

    let chi_square = observed
        .iter()
        .zip(expected.iter())
        .fold(T::zero(), |acc, (&obs, &exp)| {
            if exp <= T::zero() {
                acc
            } else {
                acc + (Float::powi(obs - exp, 2) / exp)
            }
        });

    let df = T::from(observed.len() - 1).unwrap();
    let p_value = calculate_chi_square_p_value(chi_square, df, alternative);

    TestResult::new(chi_square, p_value)
}

/// Exact binomial test for a count of successes against a hypothesised probability.
///
/// The two-sided p-value sums the probability of every outcome no more likely than
/// the observed one, matching R's `binom.test`.
pub fn binomial_test<T>(
    successes: usize,
    trials: usize,
    probability: T,
    alternative: Alternative,
) -> TestResult<T>
where
    T: FloatOpsTS,
{
    use statrs::distribution::Binomial;

    let p = probability.to_f64().unwrap_or(f64::NAN);
    if trials == 0 || successes > trials || !(0.0..=1.0).contains(&p) || !p.is_finite() {
        return TestResult::new(T::zero(), T::one());
    }

    let dist = match Binomial::new(p, trials as u64) {
        Ok(d) => d,
        Err(_) => return TestResult::new(T::zero(), T::one()),
    };

    let k = successes as u64;
    let p_value: f64 = match alternative {
        Alternative::Less => dist.cdf(k),
        Alternative::Greater => {
            if k == 0 {
                1.0
            } else {
                1.0 - dist.cdf(k - 1)
            }
        }
        Alternative::TwoSided => {
            // Sum the mass of every outcome at most as probable as the observed one.
            let observed_pmf = dist.pmf(k);
            let tol = 1e-7 * observed_pmf.max(f64::MIN_POSITIVE);
            (0..=trials as u64)
                .map(|i| dist.pmf(i))
                .filter(|&pmf| pmf <= observed_pmf + tol)
                .sum()
        }
    };

    TestResult::new(
        T::from(successes as f64).unwrap(),
        T::from(p_value.clamp(0.0, 1.0)).unwrap(),
    )
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

/// Negative binomial Wald test for one gene's counts across two groups.
///
/// Models overdispersed count data as `Var = mu + alpha * mu^2` and tests
/// `H0: log fold change = 0` with a Wald statistic. Dispersion `alpha` is estimated
/// per gene by method of moments from the pooled groups.
///
/// # Scope
///
/// This is the single-gene moment estimator, not the shrinkage/empirical-Bayes
/// machinery of DESeq2 or edgeR. It is well behaved when both groups have a
/// reasonable number of cells (pseudobulk replicates, or large clusters) and becomes
/// unstable for very small groups, where the moment estimate of `alpha` is noisy.
/// For a handful of replicates per group, prefer a dedicated bulk-RNA tool.
pub fn negative_binomial_test<T>(
    group1_counts: &[T],
    group2_counts: &[T],
    alternative: Alternative,
) -> TestResult<T>
where
    T: FloatOpsTS,
{
    let n1 = group1_counts.len() as f64;
    let n2 = group2_counts.len() as f64;
    if n1 < 2.0 || n2 < 2.0 {
        return TestResult::new(T::zero(), T::one());
    }

    let to_f64 = |v: &T| v.to_f64().unwrap_or(0.0);
    let mean1 = group1_counts.iter().map(to_f64).sum::<f64>() / n1;
    let mean2 = group2_counts.iter().map(to_f64).sum::<f64>() / n2;

    if mean1 <= 0.0 && mean2 <= 0.0 {
        return TestResult::new(T::zero(), T::one());
    }

    // Two-pass sample variances.
    let var1 = group1_counts
        .iter()
        .map(|v| (to_f64(v) - mean1).powi(2))
        .sum::<f64>()
        / (n1 - 1.0);
    let var2 = group2_counts
        .iter()
        .map(|v| (to_f64(v) - mean2).powi(2))
        .sum::<f64>()
        / (n2 - 1.0);

    // Pooled method-of-moments dispersion: alpha = (Var - mu) / mu^2, floored at 0
    // because a sub-Poisson sample variance implies no overdispersion to model.
    let pooled_mean = (mean1 * n1 + mean2 * n2) / (n1 + n2);
    let pooled_var = ((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / (n1 + n2 - 2.0);
    let alpha = if pooled_mean > 0.0 {
        ((pooled_var - pooled_mean) / (pooled_mean * pooled_mean)).max(0.0)
    } else {
        0.0
    };

    // Delta-method variance of log(mean) for an NB mean:
    //   Var(log mu_hat) ~ (mu + alpha mu^2) / (n mu^2) = 1/(n mu) + alpha/n
    let eps = 1e-10;
    let se_sq_1 = 1.0 / (n1 * mean1.max(eps)) + alpha / n1;
    let se_sq_2 = 1.0 / (n2 * mean2.max(eps)) + alpha / n2;
    let se = (se_sq_1 + se_sq_2).sqrt();

    let log_fc = (mean1.max(eps) / mean2.max(eps)).ln();
    let z = if se > 0.0 { log_fc / se } else { 0.0 };

    let p_value = if !z.is_finite() {
        1.0
    } else {
        match Normal::new(0.0, 1.0) {
            Ok(dist) => match alternative {
                Alternative::TwoSided => (2.0 * (1.0 - dist.cdf(z.abs()))).clamp(0.0, 1.0),
                Alternative::Greater => 1.0 - dist.cdf(z),
                Alternative::Less => dist.cdf(z),
            },
            Err(_) => 1.0,
        }
    };

    TestResult::new(
        T::from(z).unwrap_or_else(T::zero),
        T::from(p_value).unwrap_or_else(T::one),
    )
    .with_metadata(
        "log2_fold_change",
        T::from(log_fc / std::f64::consts::LN_2).unwrap_or_else(T::zero),
    )
    .with_metadata("dispersion", T::from(alpha).unwrap_or_else(T::zero))
}

/// Negative binomial Wald test across all genes in a sparse matrix.
///
/// See [`negative_binomial_test`] for the model and its limitations. Unstored entries
/// are materialised as zero counts, which is the correct reading for count matrices.
pub fn negative_binomial_sparse<T, N, I>(
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
    if group1_indices.is_empty() || group2_indices.is_empty() {
        return Err(anyhow::anyhow!(
            "Single-Statistics | Group indices cannot be empty. Error code: SS-NB-001"
        ));
    }

    let results = (0..matrix.n_rows)
        .into_par_iter()
        .map(|row| {
            let g1: Vec<T> = group1_indices
                .iter()
                .map(|&c| matrix.get_entry(row, c))
                .collect();
            let g2: Vec<T> = group2_indices
                .iter()
                .map(|&c| matrix.get_entry(row, c))
                .collect();
            negative_binomial_test(&g1, &g2, alternative)
        })
        .collect();

    Ok(results)
}

/// Two-part (hurdle) test for zero-inflated data.
///
/// Splits the signal in two: whether a gene is detected at all (Fisher on the
/// detection rates) and how much it is expressed where it *is* detected (Welch on the
/// non-zeros only). The two p-values are combined with Fisher's method. Same shape as
/// MAST's hurdle model — dropout is modelled rather than assumed away.
///
/// Either part is skipped when it carries no information (nothing detected, everything
/// detected, or fewer than two expressing cells in a group); the combined degrees of
/// freedom shrink accordingly. If neither part is testable the result is `p = 1`.
///
/// Reports the combined chi-square as its statistic, with both component p-values and
/// the detection rates in metadata.
pub fn zero_inflated_test<T>(
    group1: &[T],
    group2: &[T],
    alternative: Alternative,
) -> TestResult<T>
where
    T: FloatOpsTS,
{
    let (n1, n2) = (group1.len(), group2.len());
    if n1 == 0 || n2 == 0 {
        return TestResult::new(T::zero(), T::one());
    }

    let expressed = |v: &[T]| -> Vec<f64> {
        v.iter()
            .filter(|x| **x != T::zero())
            .map(|x| x.to_f64().unwrap_or(0.0))
            .collect()
    };
    let x = expressed(group1);
    let y = expressed(group2);
    let (k1, k2) = (x.len(), y.len());

    // Detection part: nothing to test if all silent or all expressed.
    let p_detect = if (k1 == 0 && k2 == 0) || (k1 == n1 && k2 == n2) {
        f64::NAN
    } else {
        fisher_exact_test::<f64>(k1, k2, n1 - k1, n2 - k2, alternative).p_value
    };

    // Expression part: needs at least two expressing cells per group.
    let p_express = if k1 < 2 || k2 < 2 {
        f64::NAN
    } else {
        crate::testing::inference::parametric::t_test(&x, &y, TTestType::Welch).p_value
    };

    // Fisher's method over whichever parts were testable.
    let (mut chi, mut df) = (0.0f64, 0.0f64);
    for p in [p_detect, p_express] {
        if p.is_finite() {
            chi += -2.0 * p.max(1e-300).ln();
            df += 2.0;
        }
    }

    let p_value = if df == 0.0 {
        1.0
    } else {
        match ChiSquared::new(df) {
            Ok(d) => (1.0 - d.cdf(chi)).clamp(0.0, 1.0),
            Err(_) => 1.0,
        }
    };

    let num = |v: f64| T::from(v).unwrap_or_else(T::zero);
    TestResult::new(num(chi), T::from(p_value).unwrap_or_else(T::one))
        .with_degrees_of_freedom(num(df))
        .with_metadata("p_detection", num(p_detect))
        .with_metadata("p_expression", num(p_express))
        .with_metadata("pct_group1", num(k1 as f64 / n1 as f64))
        .with_metadata("pct_group2", num(k2 as f64 / n2 as f64))
}

/// [`zero_inflated_test`] across all genes in a sparse matrix.
pub fn zero_inflated_sparse<T, N, I>(
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
    if group1_indices.is_empty() || group2_indices.is_empty() {
        return Err(anyhow::anyhow!(
            "Single-Statistics | Group indices cannot be empty. Error code: SS-ZI-001"
        ));
    }

    let results = (0..matrix.n_rows)
        .into_par_iter()
        .map(|row| {
            let g1: Vec<T> = group1_indices
                .iter()
                .map(|&c| matrix.get_entry(row, c))
                .collect();
            let g2: Vec<T> = group2_indices
                .iter()
                .map(|&c| matrix.get_entry(row, c))
                .collect();
            zero_inflated_test(&g1, &g2, alternative)
        })
        .collect();

    Ok(results)
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
                // A stored entry is not necessarily a non-zero one: CSR matrices
                // routinely carry explicit zeros after filtering or normalization.
                // Counting those as "expressed" collapses the contingency table.
                if matrix.val[i] == T::zero() {
                    continue;
                }
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
