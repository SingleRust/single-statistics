//! Effect size calculations for differential expression.
//!
//! # Orientation
//!
//! Every function here takes a gene (major axis) index and two sets of **cell**
//! indices along the minor axis, matching the rest of the crate. Prior to the sprs
//! migration `calculate_log2_fold_change` interpreted its index arguments as rows
//! (cells) while `calculate_cohens_d` interpreted them as columns (genes); the two
//! now agree.

use crate::testing::utils::SparseMatrixRef;
use num_traits::AsPrimitive;
use single_utilities::traits::{FloatOps, FloatOpsTS};

/// Collect one gene's values across a set of cells, materialising implicit zeros.
fn gather_gene_values<T, N, I>(
    matrix: &SparseMatrixRef<T, N, I>,
    gene: usize,
    cell_indices: &[usize],
) -> Vec<T>
where
    T: FloatOpsTS,
    N: AsPrimitive<usize> + Send + Sync,
    I: AsPrimitive<usize> + Send + Sync,
{
    cell_indices
        .iter()
        .map(|&cell| matrix.get_entry(gene, cell))
        .collect()
}

/// Calculate the log2 fold change of one gene between two groups of cells.
///
/// `pseudo_count` is added to both group means before the ratio is taken, which
/// keeps the result finite when a group has no expression at all.
pub fn calculate_log2_fold_change<T, N, I>(
    matrix: &SparseMatrixRef<T, N, I>,
    gene: usize,
    group1_indices: &[usize], // Group of interest
    group2_indices: &[usize], // Reference group
    pseudo_count: T,
) -> anyhow::Result<T>
where
    T: FloatOpsTS,
    N: AsPrimitive<usize> + Send + Sync,
    I: AsPrimitive<usize> + Send + Sync,
{
    if group1_indices.is_empty() || group2_indices.is_empty() {
        return Err(anyhow::anyhow!("Group indices cannot be empty"));
    }

    let mut sum1 = T::zero();
    for &cell in group1_indices {
        sum1 += matrix.get_entry(gene, cell);
    }

    let mut sum2 = T::zero();
    for &cell in group2_indices {
        sum2 += matrix.get_entry(gene, cell);
    }

    let n1 = T::from(group1_indices.len()).unwrap();
    let n2 = T::from(group2_indices.len()).unwrap();

    let mean1 = sum1 / n1 + pseudo_count;
    let mean2 = sum2 / n2 + pseudo_count;

    Ok((mean1 / mean2).log2())
}

/// Calculate Cohen's d effect size for one gene between two groups of cells.
pub fn calculate_cohens_d<T, N, I>(
    matrix: &SparseMatrixRef<T, N, I>,
    gene: usize,
    group1_indices: &[usize],
    group2_indices: &[usize],
) -> anyhow::Result<T>
where
    T: FloatOpsTS,
    N: AsPrimitive<usize> + Send + Sync,
    I: AsPrimitive<usize> + Send + Sync,
{
    if group1_indices.len() < 2 || group2_indices.len() < 2 {
        return Err(anyhow::anyhow!(
            "Each group must have at least 2 samples for Cohen's d"
        ));
    }

    let group1_values = gather_gene_values(matrix, gene, group1_indices);
    let group2_values = gather_gene_values(matrix, gene, group2_indices);

    let n1 = T::from(group1_values.len()).unwrap();
    let n2 = T::from(group2_values.len()).unwrap();

    let mean1 = group1_values.iter().copied().sum::<T>() / n1;
    let mean2 = group2_values.iter().copied().sum::<T>() / n2;

    // Two-pass variance: numerically stable where the sum-of-squares shortcut is not.
    let var1 = group1_values
        .iter()
        .map(|&x| num_traits::Float::powi(x - mean1, 2))
        .sum::<T>()
        / (n1 - T::one());

    let var2 = group2_values
        .iter()
        .map(|&x| num_traits::Float::powi(x - mean2, 2))
        .sum::<T>()
        / (n2 - T::one());

    let pooled_sd = (((n1 - T::one()) * var1 + (n2 - T::one()) * var2)
        / (n1 + n2 - T::from(2.0).unwrap()))
    .sqrt();

    Ok((mean2 - mean1) / pooled_sd)
}

/// Calculate Hedges' g, the bias-corrected form of Cohen's d.
pub fn calculate_hedges_g<T, N, I>(
    matrix: &SparseMatrixRef<T, N, I>,
    gene: usize,
    group1_indices: &[usize],
    group2_indices: &[usize],
) -> anyhow::Result<T>
where
    T: FloatOpsTS,
    N: AsPrimitive<usize> + Send + Sync,
    I: AsPrimitive<usize> + Send + Sync,
{
    let d = calculate_cohens_d(matrix, gene, group1_indices, group2_indices)?;

    let one = T::one();
    let two = T::from(2.0).unwrap();
    let three = T::from(3.0).unwrap();
    let four = T::from(4.0).unwrap();

    let n1 = T::from(group1_indices.len()).unwrap();
    let n2 = T::from(group2_indices.len()).unwrap();
    let n = n1 + n2;

    // Correction factor J
    let j = one - three / (four * (n - two) - one);

    Ok(j * d)
}

/// Cohen's d from two already-materialised samples.
///
/// Useful when the values have been gathered by other means (for example from a
/// dense array or a pseudobulk aggregation).
pub fn cohens_d_from_samples<T>(x: &[T], y: &[T]) -> anyhow::Result<T>
where
    T: FloatOps,
{
    if x.len() < 2 || y.len() < 2 {
        return Err(anyhow::anyhow!(
            "Each group must have at least 2 samples for Cohen's d"
        ));
    }

    let n1 = T::from(x.len()).unwrap();
    let n2 = T::from(y.len()).unwrap();

    let mean1 = x.iter().copied().sum::<T>() / n1;
    let mean2 = y.iter().copied().sum::<T>() / n2;

    let var1 = x
        .iter()
        .map(|&v| num_traits::Float::powi(v - mean1, 2))
        .sum::<T>()
        / (n1 - T::one());
    let var2 = y
        .iter()
        .map(|&v| num_traits::Float::powi(v - mean2, 2))
        .sum::<T>()
        / (n2 - T::one());

    let pooled_sd = (((n1 - T::one()) * var1 + (n2 - T::one()) * var2)
        / (n1 + n2 - T::from(2.0).unwrap()))
    .sqrt();

    Ok((mean2 - mean1) / pooled_sd)
}
