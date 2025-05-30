use nalgebra_sparse::CsrMatrix;
use single_utilities::traits::FloatOpsTS;

pub fn extract_unique_groups(group_ids: &[usize]) -> Vec<usize> {
    let mut unique_groups = group_ids.to_vec();
    unique_groups.sort();
    unique_groups.dedup();
    unique_groups
}

/// Get indices for each group
pub fn get_group_indices(group_ids: &[usize], unique_groups: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let group1 = unique_groups[0];
    let group2 = unique_groups[1];

    let group1_indices = group_ids.iter()
        .enumerate()
        .filter_map(|(i, &g)| if g == group1 { Some(i) } else { None })
        .collect();

    let group2_indices = group_ids.iter()
        .enumerate()
        .filter_map(|(i, &g)| if g == group2 { Some(i) } else { None })
        .collect();

    (group1_indices, group2_indices)
}

pub(crate) fn accumulate_gene_statistics_two_groups<T>(
    matrix: &CsrMatrix<T>,
    group1_indices: &[usize],
    group2_indices: &[usize],
) -> anyhow::Result<(Vec<T>, Vec<T>, Vec<T>, Vec<T>)>
where
    T: FloatOpsTS,
{
    let n_genes = matrix.ncols();

    // Pre-allocate all accumulation vectors
    let mut group1_sums = vec![T::zero(); n_genes];
    let mut group1_sum_squares = vec![T::zero(); n_genes];
    let mut group2_sums = vec![T::zero(); n_genes];
    let mut group2_sum_squares = vec![T::zero(); n_genes];
    
    for &row_idx in group1_indices {
        let row = matrix.row(row_idx);
        for (col_idx, &value) in row.col_indices().iter().zip(row.values().iter()) {
            group1_sums[*col_idx] += value;
            group1_sum_squares[*col_idx] += value * value;
        }
    }
    
    for &row_idx in group2_indices {
        let row = matrix.row(row_idx);
        for (col_idx, &value) in row.col_indices().iter().zip(row.values().iter()) {
            group2_sums[*col_idx] += value;
            group2_sum_squares[*col_idx] += value * value;
        }
    }

    Ok((group1_sums, group1_sum_squares, group2_sums, group2_sum_squares))
}

pub(crate) fn extract_gene_values_optimized<T>(
    matrix: &CsrMatrix<T>,
    gene_idx: usize,
    group1_indices: &[usize],
    group2_indices: &[usize],
) -> (Vec<T>, Vec<T>)
where
    T: FloatOpsTS,
{
    // Pre-allocate with known capacity
    let mut group1_values = Vec::with_capacity(group1_indices.len());
    let mut group2_values = Vec::with_capacity(group2_indices.len());

    // Extract values efficiently
    for &row_idx in group1_indices {
        let value = matrix.get_entry(row_idx, gene_idx)
            .map(|entry| entry.into_value())
            .unwrap_or(T::zero());
        group1_values.push(value);
    }

    for &row_idx in group2_indices {
        let value = matrix.get_entry(row_idx, gene_idx)
            .map(|entry| entry.into_value())
            .unwrap_or(T::zero());
        group2_values.push(value);
    }

    (group1_values, group2_values)
}