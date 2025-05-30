use crate::testing::FloatOps;
use nalgebra_sparse::CsrMatrix;
use num_traits::{Float, FromPrimitive, NumCast};
use single_utilities::traits::FloatOpsTS;
use std::fmt::Debug;

/// Calculate log2 fold change between two groups
pub fn calculate_log2_fold_change<T>(
    matrix: &CsrMatrix<T>,
    col: usize,
    group1_indices: &[usize], // Group of interest
    group2_indices: &[usize], // Reference group
    pseudo_count: T,        // Small value like 1e-9 or 1.0
) -> anyhow::Result<T>
where
    T: FloatOps,
{
    if group1_indices.is_empty() || group2_indices.is_empty() {
        return Err(anyhow::anyhow!("Group indices cannot be empty"));
    }

    let mut sum1 = T::zero();
    let mut count1 = 0;
    for &row in group1_indices {
        if let Some(entry) = matrix.get_entry(row, col) {
            let value = entry.into_value();
            sum1 += value;
            count1 += 1;
        }
    }

    let mut sum2 = T::zero();
    let mut count2 = 0;
    for &row in group2_indices {
        if let Some(entry) = matrix.get_entry(row, col) {
            let value = entry.into_value();
            sum2 += value;
            count2 += 1;
        }
    }
    let go_f64 = T::from(group1_indices.len()).unwrap();
    let gt_f64 = T::from(group2_indices.len()).unwrap();

    let mean1 = sum1 / go_f64 + pseudo_count;
    let mean2 = sum2 / gt_f64 + pseudo_count;

    let log2_fc = (mean1 / mean2).log2();

    Ok(log2_fc)
}

/// Calculate Cohen's d effect size for a row
pub fn calculate_cohens_d<T>(
    matrix: &CsrMatrix<T>,
    row: usize,
    group1_indices: &[usize],
    group2_indices: &[usize],
) -> anyhow::Result<T>
where
    T: FloatOpsTS,
{
    if group1_indices.len() < 2 || group2_indices.len() < 2 {
        return Err(anyhow::anyhow!(
            "Each group must have at least 2 samples for Cohen's d"
        ));
    }

    // Extract values for group 1
    let mut sum_g1 = T::zero();
    let mut group1_values = Vec::with_capacity(group1_indices.len());
    for &col in group1_indices {
        if let Some(entry) = matrix.get_entry(row, col) {
            let value = entry.into_value();
            sum_g1 += value;
            group1_values.push(value);
        } else {
            group1_values.push(T::zero());
        }
    }

    // Extract values for group 2
    let mut sum_g2 = T::zero();
    let mut group2_values = Vec::with_capacity(group2_indices.len());
    for &col in group2_indices {
        if let Some(entry) = matrix.get_entry(row, col) {
            let value = entry.into_value();
            sum_g2 += value;
            group2_values.push(value);
        } else {
            group2_values.push(T::zero());
        }
    }

    let go_t = T::from(group1_indices.len()).unwrap();
    let gt_t = T::from(group2_indices.len()).unwrap();

    // Calculate means
    let mean1 = sum_g1 / go_t;
    let mean2 = sum_g2 / gt_t;
    // Calculate variances
    let var1 = group1_values
        .iter()
        .map(|&x| num_traits::Float::powi((x - mean1), 2))
        .sum::<T>()
        / (go_t - T::one());

    let var2 = group2_values
        .iter()
        .map(|&x| num_traits::Float::powi((x - mean2), 2))
        .sum::<T>()
        / (gt_t - T::one());

    // Calculate pooled standard deviation
    let pooled_sd = (((go_t - T::one()) * var1 + (gt_t - T::one()) * var2) / (go_t + gt_t - T::from(2.0).unwrap())).sqrt();

    // Calculate Cohen's d
    let d = (mean2 - mean1) / pooled_sd;

    Ok(d)
}

/// Calculate Hedge's g (bias-corrected effect size)
pub fn calculate_hedges_g<T>(
    matrix: &CsrMatrix<T>,
    row: usize,
    group1_indices: &[usize],
    group2_indices: &[usize],
) -> anyhow::Result<T>
where
    T: FloatOpsTS,
{
    // First calculate Cohen's d
    let d = calculate_cohens_d(matrix, row, group1_indices, group2_indices)?;
    let one = T::one();
    let two = T::from(2.0).unwrap();
    let three = T::from(3.0).unwrap();
    let four = T::from(4.0).unwrap();
    // Apply correction factor
    let n1 = T::from(group1_indices.len()).unwrap();
    let n2 = T::from(group2_indices.len()).unwrap();
    let n = n1 + n2;

    // Correction factor J
    let j = one - three / (four * (n - two) - one);

    // Calculate Hedge's g
    let g = j * d;

    Ok(g)
}


