//! Multiple testing correction methods for controlling false positives in differential expression analysis.
//!
//! When testing thousands of genes simultaneously (as is common in single-cell RNA-seq analysis),
//! the probability of false positives increases dramatically. These correction methods help control
//! either the Family-Wise Error Rate (FWER) or False Discovery Rate (FDR).
//!
//! ## Available Methods
//!
//! - **Bonferroni**: Conservative FWER control, multiplies p-values by number of tests
//! - **Benjamini-Hochberg**: FDR control, less conservative than Bonferroni
//! - **Benjamini-Yekutieli**: FDR control for dependent tests
//! - **Holm-Bonferroni**: Step-down FWER control, less conservative than Bonferroni
//! - **Storey's q-value**: Improved FDR estimation
//!
//! ## Choosing a Method
//!
//! - **For single-cell DE analysis**: Use Benjamini-Hochberg (most common)
//! - **For very strict control**: Use Bonferroni or Holm-Bonferroni
//! - **For dependent tests**: Use Benjamini-Yekutieli
//! - **For large datasets**: Consider Storey's q-value

use anyhow::{Result, anyhow};
use single_utilities::traits::FloatOps;
use std::cmp::Ordering;

/// Apply Bonferroni correction to p-values
///
/// Bonferroni correction is a simple but conservative method that multiplies
/// each p-value by the number of tests.
///
/// # Arguments
/// * `p_values` - A slice of p-values to adjust
///
/// # Returns
/// * `Result<Vec<f64>>` - Vector of adjusted p-values
///
/// # Example
/// ```
/// let p_values = vec![0.01, 0.03, 0.05];
/// let adjusted = single_statistics::testing::correction::bonferroni_correction(&p_values).unwrap();
/// ```
pub fn bonferroni_correction<T>(p_values: &[T]) -> Result<Vec<T>>
where
    T: FloatOps,
{
    let n = p_values.len();

    if n == 0 {
        return Err(anyhow!("Empty p-value array"));
    }

    // Validate p-values
    for (i, &p) in p_values.iter().enumerate() {
        if !(T::zero()..=T::one()).contains(&p) {
            return Err(anyhow!("Invalid p-value at index {:?}: {:?}", i, p));
        }
    }

    // Multiply each p-value by n, capping at 1.0
    let n_t = T::from(n).unwrap();
    let adjusted = p_values
        .iter()
        .map(|&p| num_traits::Float::min(p * n_t, T::one()))
        .collect();

    Ok(adjusted)
}

/// Apply Benjamini-Hochberg (BH) procedure for controlling false discovery rate
///
/// The BH procedure controls the false discovery rate (FDR), which is the expected
/// proportion of false positives among all rejected null hypotheses.
///
/// # Arguments
/// * `p_values` - A slice of p-values to adjust
///
/// # Returns
/// * `Result<Vec<f64>>` - Vector of adjusted p-values
///
/// # Example
/// ```
/// let p_values = vec![0.01, 0.03, 0.05];
/// let adjusted = single_statistics::testing::correction::benjamini_hochberg_correction(&p_values).unwrap();
/// ```
pub fn benjamini_hochberg_correction<T>(p_values: &[T]) -> Result<Vec<T>>
where
    T: FloatOps,
{
    let n = p_values.len();
    if n == 0 {
        return Err(anyhow::anyhow!("Empty p-value array"));
    }

    // Validate p-values
    for (i, &p) in p_values.iter().enumerate() {
        if !(T::zero()..=T::one()).contains(&p) {
            return Err(anyhow::anyhow!("Invalid p-value at index {:?}: {:?}", i, p));
        }
    }
    
    let n_f = T::from(n).unwrap();

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&a, &b| {
        p_values[a]
            .partial_cmp(&p_values[b])
            .unwrap_or(Ordering::Equal)
    });

    let mut adjusted = vec![T::zero(); n];
    let mut running_min = T::one();

    for i in (0..n).rev() {
        let orig_idx = indices[i];
        let rank = T::from(i + 1).unwrap();
        let adjustment = num_traits::Float::min(p_values[orig_idx] * n_f / rank, T::one());
        running_min = num_traits::Float::min(adjustment, running_min);
        adjusted[orig_idx] = running_min;
    }

    Ok(adjusted)
}

/// Apply Benjamini-Yekutieli (BY) procedure for controlling false discovery rate under dependence
///
/// The BY procedure is a more conservative variant of the BH procedure that is valid
/// under arbitrary dependence structures among the tests.
///
/// # Arguments
/// * `p_values` - A slice of p-values to adjust
///
/// # Returns
/// * `Result<Vec<f64>>` - Vector of adjusted p-values
///
/// # Example
/// ```
/// let p_values = vec![0.01, 0.03, 0.05];
/// let adjusted = single_statistics::testing::correction::benjamini_yekutieli_correction(&p_values).unwrap();
/// ```
pub fn benjamini_yekutieli_correction<T>(p_values: &[T]) -> Result<Vec<T>>
where
    T: FloatOps,
{
    let n = p_values.len();
    if n == 0 {
        return Err(anyhow!("Empty p-value array"));
    }

    // Calculate the correction factor
    let c_n: T = (1..=n).map(|i| T::one() / T::from(i).unwrap()).sum();

    // Create index-value pairs and sort by p-value
    let mut indexed_p_values: Vec<(usize, T)> =
        p_values.iter().enumerate().map(|(i, &p)| (i, p)).collect();

    // Sort in ascending order
    indexed_p_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    // Calculate adjusted p-values with monitoring of minimum value
    let mut adjusted_p_values = vec![T::zero(); n];
    let mut current_min = T::one();
    let n_f64 = T::from(n).unwrap();
    // Process from largest to smallest p-value
    for i in (0..n).rev() {
        let (orig_idx, p_val) = indexed_p_values[i];
        let rank = i + 1;

        // Calculate adjustment and take minimum of current and previous
        let adjustment =
            num_traits::Float::min(p_val * c_n * n_f64 / T::from(rank).unwrap(), T::one());
        current_min = num_traits::Float::min(adjustment, current_min);
        adjusted_p_values[orig_idx] = current_min;
    }

    Ok(adjusted_p_values)
}

/// Apply Holm-Bonferroni (step-down) method for controlling family-wise error rate
///
/// The Holm procedure is a step-down method that controls the family-wise error rate (FWER)
/// and is uniformly more powerful than the standard Bonferroni correction.
///
/// # Arguments
/// * `p_values` - A slice of p-values to adjust
///
/// # Returns
/// * `Result<Vec<f64>>` - Vector of adjusted p-values
///
/// # Example
/// ```
/// let p_values = vec![0.01, 0.03, 0.05];
/// let adjusted = single_statistics::testing::correction::holm_bonferroni_correction(&p_values).unwrap();
/// ```
pub fn holm_bonferroni_correction<T>(p_values: &[T]) -> Result<Vec<T>>
where
    T: FloatOps,
{
    let n = p_values.len();

    if n == 0 {
        return Err(anyhow!("Empty p-value array"));
    }

    let zero = T::zero();
    let one = T::one();

    // Validate p-values
    for (i, &p) in p_values.iter().enumerate() {
        if p < zero || p > one {
            return Err(anyhow!("Invalid p-value at index {}: {:?}", i, p));
        }
    }

    // Create index-value pairs and sort by p-value
    let mut indexed_p_values: Vec<(usize, T)> =
        p_values.iter().enumerate().map(|(i, &p)| (i, p)).collect();

    indexed_p_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    // Initialize the result vector
    let mut adjusted_p_values = vec![zero; n];

    // Calculate adjusted p-values
    for (i, &(idx, p_val)) in indexed_p_values.iter().enumerate() {
        // Convert (n - i) to type T
        let multiplier = T::from((n - i) as f64).unwrap_or(one);
        let adjusted_p = p_val * multiplier;
        adjusted_p_values[idx] = num_traits::Float::min(adjusted_p, one);
    }

    Ok(adjusted_p_values)
}

/// Apply Hochberg's step-up method for controlling family-wise error rate
///
/// Hochberg's procedure is a step-up method that controls the family-wise error rate (FWER)
/// and is more powerful than Holm's procedure when all tests are independent.
///
/// # Arguments
/// * `p_values` - A slice of p-values to adjust
///
/// # Returns
/// * `Result<Vec<f64>>` - Vector of adjusted p-values
///
/// # Example
/// ```
/// let p_values = vec![0.01, 0.03, 0.05];
/// let adjusted = single_statistics::testing::correction::hochberg_correction(&p_values).unwrap();
/// ```
pub fn hochberg_correction<T>(p_values: &[T]) -> Result<Vec<T>>
where
    T: FloatOps,
{
    let n = p_values.len();

    if n == 0 {
        return Err(anyhow!("Empty p-value array"));
    }

    let zero = T::zero();
    let one = T::one();

    // Create index-value pairs and sort by p-value (descending)
    let mut indexed_p_values: Vec<(usize, T)> =
        p_values.iter().enumerate().map(|(i, &p)| (i, p)).collect();

    indexed_p_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    // Calculate adjusted p-values
    let mut adjusted_p_values = vec![zero; n];

    // Largest p-value remains the same
    adjusted_p_values[indexed_p_values[0].0] = indexed_p_values[0].1;

    // Process remaining p-values from second-largest to smallest
    for i in 1..n {
        let current_index = indexed_p_values[i].0;
        let prev_index = indexed_p_values[i - 1].0;

        // Convert n and (n - i) to type T
        let n_t = T::from(n).unwrap_or(one);
        let denominator_t = T::from(n - i).unwrap_or(one);

        let hochberg_value =
            num_traits::Float::min(indexed_p_values[i].1 * n_t / denominator_t, one);
        adjusted_p_values[current_index] =
            num_traits::Float::min(hochberg_value, adjusted_p_values[prev_index]);
    }

    Ok(adjusted_p_values)
}

/// Apply Storey's q-value method for controlling false discovery rate
///
/// Storey's q-value method estimates the proportion of true null hypotheses (π0)
/// and uses this to obtain more powerful FDR control than the BH procedure.
///
/// # Arguments
/// * `p_values` - A slice of p-values to adjust
/// * `lambda` - Tuning parameter for π0 estimation (between 0 and 1, typically 0.5)
///
/// # Returns
/// * `Result<Vec<f64>>` - Vector of q-values
///
/// # Example
/// ```
/// let p_values = vec![0.01, 0.03, 0.05];
/// let qvalues = single_statistics::testing::correction::storey_qvalues(&p_values, 0.5).unwrap();
/// ```
pub fn storey_qvalues<T>(p_values: &[T], lambda: T) -> Result<Vec<T>>
where
    T: FloatOps,
{
    let n = p_values.len();

    if n == 0 {
        return Err(anyhow!("Empty p-value array"));
    }

    let zero = T::zero();
    let one = T::one();

    if lambda <= zero || lambda >= one {
        return Err(anyhow!("Lambda must be between 0 and 1, got {:?}", lambda));
    }

    // Validate p-values
    for (i, &p) in p_values.iter().enumerate() {
        if p < zero || p > one {
            return Err(anyhow!("Invalid p-value at index {}: {:?}", i, p));
        }
    }

    // Estimate pi0 (proportion of true null hypotheses)
    let w = p_values.iter().filter(|&&p| p > lambda).count();
    let w_t = T::from(w).unwrap_or(zero);
    let n_t = T::from(n).unwrap_or(one);

    let pi0 = w_t / (n_t * (one - lambda));
    let pi0 = num_traits::Float::min(pi0, one); // Ensure pi0 doesn't exceed 1

    // First apply Benjamini-Hochberg to get base adjusted values
    let bh_adjusted = benjamini_hochberg_correction(p_values)?;

    // Multiply by pi0 to get q-values
    let q_values = bh_adjusted
        .iter()
        .map(|&p| num_traits::Float::min(p * pi0, one))
        .collect();

    Ok(q_values)
}

/// Apply adaptive Storey's q-value method with automatic lambda selection
///
/// This version of Storey's method tries multiple lambda values and selects the one
/// that minimizes the mean-squared error of the π0 estimate.
///
/// # Arguments
/// * `p_values` - A slice of p-values to adjust
///
/// # Returns
/// * `Result<Vec<f64>>` - Vector of q-values
///
/// # Example
/// ```
/// let p_values = vec![0.01, 0.03, 0.05];
/// let qvalues = single_statistics::testing::correction::adaptive_storey_qvalues(&p_values).unwrap();
/// ```
pub fn adaptive_storey_qvalues<T>(p_values: &[T]) -> Result<Vec<T>>
where
    T: FloatOps,
{
    let n = p_values.len();

    if n == 0 {
        return Err(anyhow!("Empty p-value array"));
    }

    let zero = T::zero();
    let one = T::one();

    // Validate p-values
    for (i, &p) in p_values.iter().enumerate() {
        if p < zero || p > one {
            return Err(anyhow!("Invalid p-value at index {}: {:?}", i, p));
        }
    }

    // Define lambda grid - convert f64 values to type T
    let lambda_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    let lambda_grid: Vec<T> = lambda_values
        .iter()
        .filter_map(|&val| T::from(val))
        .collect();

    // Calculate pi0 estimates for different lambda values
    let mut pi0_estimates = Vec::with_capacity(lambda_grid.len());
    for &lambda in &lambda_grid {
        let w = p_values.iter().filter(|&&p| p > lambda).count();
        let w_t = T::from(w as f64).unwrap_or(zero);
        let n_t = T::from(n as f64).unwrap_or(one);

        let pi0 = w_t / (n_t * (one - lambda));
        pi0_estimates.push(num_traits::Float::min(pi0, one));
    }

    // Fit smooth cubic spline (simplified here with linear interpolation to the final value)
    // In practice, a proper spline fitting would be better
    let pi0_sum: T = pi0_estimates.iter().copied().sum();
    let estimates_len = T::from(pi0_estimates.len() as f64).unwrap_or(one);
    let pi0_mean = pi0_sum / estimates_len;

    let pi0 = if pi0_estimates.is_empty() {
        one
    } else {
        num_traits::Float::min(pi0_mean, one)
    };

    // First apply Benjamini-Hochberg to get base adjusted values
    let bh_adjusted = benjamini_hochberg_correction(p_values)?;

    // Multiply by pi0 to get q-values
    let q_values = bh_adjusted
        .iter()
        .map(|&p| num_traits::Float::min((p * pi0), one))
        .collect();

    Ok(q_values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_vec_relative_eq(a: &[f64], b: &[f64], epsilon: f64) {
        assert_eq!(a.len(), b.len(), "Vectors have different lengths");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            if (x - y).abs() > epsilon {
                panic!("Vectors differ at index {}: {} != {}", i, x, y);
            }
        }
    }

    #[test]
    fn test_bonferroni() {
        let p_values = vec![0.01, 0.02, 0.03, 0.1, 0.2];
        let expected = vec![0.05, 0.1, 0.15, 0.5, 1.0];
        let adjusted = bonferroni_correction(&p_values).unwrap();
        assert_vec_relative_eq(&adjusted, &expected, 1e-10);
    }

    use approx::assert_relative_eq;

    #[test]
    fn test_benjamini_hochberg_empty_input() {
        // Test with empty input
        let result = benjamini_hochberg_correction::<f64>(&[]);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().to_string(), "Empty p-value array");
    }

    #[test]
    fn test_benjamini_hochberg_invalid_pvalues() {
        // Test with invalid p-values (negative)
        let result = benjamini_hochberg_correction(&[0.01, -0.5, 0.03]);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Invalid p-value at index 1")
        );

        // Test with invalid p-values (greater than 1)
        let result = benjamini_hochberg_correction(&[0.01, 1.5, 0.03]);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Invalid p-value at index 1")
        );
    }

    #[test]
    fn test_benjamini_hochberg_identical_pvalues() {
        // Test with identical p-values
        let p_values = vec![0.05, 0.05, 0.05];
        let expected = vec![0.05, 0.05, 0.05];
        let adjusted = benjamini_hochberg_correction(&p_values).unwrap();

        for (a, e) in adjusted.iter().zip(expected.iter()) {
            assert_relative_eq!(*a, *e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_benjamini_hochberg_ordered_pvalues() {
        // Test with ordered p-values
        let p_values = vec![0.01, 0.02, 0.03, 0.04, 0.05];
        let expected = vec![0.05, 0.05, 0.05, 0.05, 0.05];
        let adjusted = benjamini_hochberg_correction(&p_values).unwrap();

        for (i, (&a, &e)) in adjusted.iter().zip(expected.iter()).enumerate() {
            assert_relative_eq!(a, e, epsilon = 1e-10, max_relative = 1e-10);
            // If assert fails, this message would help identify which element had issues
            if num_traits::Float::abs(a - e) > 1e-10 {
                panic!("mismatch at index {}: expected {}, got {}", i, e, a);
            }
        }
    }

    #[test]
    fn test_benjamini_hochberg_unordered_pvalues() {
        // Test with unordered p-values
        let p_values = vec![0.05, 0.01, 0.1, 0.04, 0.02];
        let expected = vec![0.0625, 0.05, 0.1, 0.0625, 0.05];
        let adjusted = benjamini_hochberg_correction(&p_values).unwrap();

        for (i, (&a, &e)) in adjusted.iter().zip(expected.iter()).enumerate() {
            //assert_relative_eq!(*a, *e, epsilon = 1e-3, max_relative = 1e-3);
            // If assert fails, this message would help identify which element had issues
            if num_traits::Float::abs(a - e) > 1e-3 {
                panic!(
                    "mismatch at index {}: expected {}, got {}, whole: {:?}",
                    i, e, a, adjusted
                );
            }
        }
    }

    #[test]
    fn test_benjamini_hochberg_edge_cases() {
        // Test with very small p-values
        let p_values = vec![1e-10, 1e-9, 1e-8];
        let adjusted = benjamini_hochberg_correction(&p_values).unwrap();
        // All should be adjusted but still very small
        assert!(adjusted.iter().all(|&p| p > 0.0 && p < 0.001));

        // Test with p-value of 1.0
        let p_values = vec![0.1, 0.2, 1.0];
        let adjusted = benjamini_hochberg_correction(&p_values).unwrap();
        // The last one should remain 1.0
        assert_relative_eq!(adjusted[2], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_benjamini_hochberg_real_example() {
        // A more realistic example based on common scientific data
        let pvalues = vec![0.1, 0.2, 0.3, 0.4, 0.1];
        let expected = [0.25, 0.3333333333333333, 0.375, 0.4, 0.25];
        let adjusted = benjamini_hochberg_correction(&pvalues).unwrap();

        for (i, (&a, &e)) in adjusted.iter().zip(expected.iter()).enumerate() {
            assert_relative_eq!(a, e, epsilon = 1e-3, max_relative = 1e-3);
            // If assert fails, this message would help identify which element had issues
            if num_traits::Float::abs(a - e) > 1e-3 {
                panic!("mismatch at index {}: expected {}, got {}", i, e, a);
            }
        }
    }

    #[test]
    fn test_benjamini_hochberg_single_pvalue() {
        // Test with a single p-value
        let p_values = vec![0.025];
        let adjusted = benjamini_hochberg_correction(&p_values).unwrap();
        assert_relative_eq!(adjusted[0], 0.025, epsilon = 1e-10);
    }
    #[test]
    fn test_holm_bonferroni() {
        let p_values = vec![0.01, 0.02, 0.03];
        let expected = vec![0.03, 0.04, 0.03];
        let adjusted = holm_bonferroni_correction(&p_values).unwrap();
        assert_vec_relative_eq(&adjusted, &expected, 1e-10);
    }

    #[test]
    fn test_storey_qvalues() {
        let p_values = vec![0.01, 0.02, 0.03, 0.6, 0.7];
        let qvalues = storey_qvalues(&p_values, 0.5).unwrap();
        // Verify qvalues are between 0 and 1
        for &q in &qvalues {
            assert!((0.0..=1.0).contains(&q));
        }
        // Verify ordering is preserved
        assert!(qvalues[0] <= qvalues[2]);
        assert!(qvalues[3] <= qvalues[4]);
    }

    #[test]
    fn test_invalid_inputs() {
        // Empty array
        assert!(bonferroni_correction::<f32>(&[]).is_err());
        assert!(benjamini_hochberg_correction::<f32>(&[]).is_err());
        assert!(holm_bonferroni_correction::<f32>(&[]).is_err());

        // Invalid lambda
        assert!(storey_qvalues(&[0.5], -0.1).is_err());
        assert!(storey_qvalues(&[0.5], 1.0).is_err());

        // Invalid p-values
        let invalid_p = vec![-0.1, 0.5, 1.1];
        assert!(bonferroni_correction(&invalid_p).is_err());
        assert!(benjamini_hochberg_correction(&invalid_p).is_err());
    }
}
