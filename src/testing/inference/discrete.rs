use crate::testing::{Alternative, TestResult};
use single_utilities::traits::FloatOps;
use statrs::distribution::{ChiSquared, ContinuousCDF};

/// Performs a chi-square test for independence on a 2x2 contingency table
pub fn chi_square_test<T>(
    a: T,
    b: T,
    c: T,
    d: T,
    alternative: Alternative,
) -> TestResult<T>
where
    T: FloatOps,
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
    let chi_square = (num_traits::Float::powi(a - expected_a, 2) / expected_a)
        + (num_traits::Float::powi(b - expected_b, 2) / expected_b)
        + (num_traits::Float::powi(c - expected_c, 2) / expected_c)
        + (num_traits::Float::powi(d - expected_d, 2) / expected_d);

    // Calculate p-value using chi-square distribution with 1 degree of freedom
    let p_value = calculate_chi_square_p_value(chi_square, T::one(), alternative);

    TestResult::new(chi_square, p_value)
}

/// Performs a chi-square test for goodness of fit
pub fn chi_square_goodness_of_fit<T>(
    observed: &[T],
    expected: &[T],
    alternative: Alternative,
) -> TestResult<T>
where
    T: FloatOps,
{
    if observed.len() != expected.len() {
        return TestResult::new(T::zero(), T::one());
    }

    let chi_square = observed
        .iter()
        .zip(expected.iter())
        .fold(T::zero(), |acc, (&obs, &exp)| {
            if exp <= T::zero() {
                acc
            } else {
                acc + (num_traits::Float::powi(obs - exp, 2) / exp)
            }
        });

    let df = T::from(observed.len() - 1).unwrap();
    let p_value = calculate_chi_square_p_value(chi_square, df, alternative);

    TestResult::new(chi_square, p_value)
}

fn calculate_chi_square_p_value<T>(chi_square: T, df: T, alternative: Alternative) -> T
where
    T: FloatOps,
{
    let chi_square_f64 = chi_square.to_f64().unwrap();
    let df_f64 = df.to_f64().unwrap();

    match ChiSquared::new(df_f64) {
        Ok(chi_dist) => {
            let p = match alternative {
                Alternative::TwoSided => 2.0 * (1.0 - chi_dist.cdf(chi_square_f64)),
                Alternative::Less => chi_dist.cdf(chi_square_f64),
                Alternative::Greater => 1.0 - chi_dist.cdf(chi_square_f64),
            };
            T::from(p).unwrap()
        }
        Err(_) => T::one(), // Fallback for invalid parameters
    }
}

/// Performs a binomial test
pub fn binomial_test<T>(
    successes: usize,
    trials: usize,
    probability: T,
    alternative: Alternative,
) -> TestResult<T>
where
    T: FloatOps,
{
    if trials == 0 || probability <= T::zero() || probability >= T::one() {
        return TestResult::new(T::zero(), T::one());
    }

    let p = T::from(probability).unwrap();
    let _q = T::one() - p;
    let n = T::from(trials).unwrap();
    let k = T::from(successes).unwrap();

    // Calculate the probability mass function for the observed value
    let observed_pmf = binomial_pmf(k, n, p);

    // Calculate p-value based on alternative hypothesis
    let p_value = match alternative {
        Alternative::TwoSided => {
            // Sum probabilities of all outcomes that are as or more extreme
            let mut sum = T::zero();
            for i in 0..=trials {
                let i_t = T::from(i).unwrap();
                let pmf = binomial_pmf(i_t, n, p);
                if pmf <= observed_pmf {
                    sum += pmf;
                }
            }
            sum
        }
        Alternative::Less => {
            // Sum probabilities of outcomes less than or equal to observed
            let mut sum = T::zero();
            for i in 0..=successes {
                let i_t = T::from(i).unwrap();
                sum += binomial_pmf(i_t, n, p);
            }
            sum
        }
        Alternative::Greater => {
            // Sum probabilities of outcomes greater than or equal to observed
            let mut sum = T::zero();
            for i in successes..=trials {
                let i_t = T::from(i).unwrap();
                sum += binomial_pmf(i_t, n, p);
            }
            sum
        }
    };

    TestResult::new(T::from(successes as f64).unwrap(), p_value)
}

fn binomial_pmf<T>(k: T, n: T, p: T) -> T
where
    T: FloatOps,
{
    let k_f64 = k.to_f64().unwrap();
    let n_f64 = n.to_f64().unwrap();
    
    // Calculate binomial coefficient
    let mut coeff = 1.0;
    for i in 0..k_f64 as usize {
        coeff *= (n_f64 - i as f64) / (i as f64 + 1.0);
    }
    
    let coeff_t = T::from(coeff).unwrap();
    coeff_t * p.powf(k) * (T::one() - p).powf(n - k)
}
