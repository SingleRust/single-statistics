//! Utility functions for statistical testing operations.

use single_utilities::traits::FloatOpsTS;
use num_traits::AsPrimitive;

/// A lightweight reference-based representation of a sparse matrix (CSR or CSC).
/// 
/// This structure is designed to be agnostic of the underlying container and can be 
/// easily used with raw vectors from other crates or FFI (like PyO3).
#[derive(Debug, Clone, Copy)]
pub struct SparseMatrixRef<'a, T, N, I> {
    /// Major indices (e.g., indptr in CSR/CSC)
    pub maj_ind: &'a [N],
    /// Minor indices (e.g., column indices in CSR, row indices in CSC)
    pub min_ind: &'a [I],
    /// The actual values in the matrix
    pub val: &'a [T],
    /// Number of rows in the matrix
    pub n_rows: usize,
    /// Number of columns in the matrix
    pub n_cols: usize,
}

impl<'a, T, N, I> SparseMatrixRef<'a, T, N, I>
where
    T: FloatOpsTS,
    N: AsPrimitive<usize> + Send + Sync,
    I: AsPrimitive<usize> + Send + Sync,
{
    /// Create a new SparseMatrixRef
    pub fn new(maj_ind: &'a [N], min_ind: &'a [I], val: &'a [T], n_rows: usize, n_cols: usize) -> Self {
        Self { maj_ind, min_ind, val, n_rows, n_cols }
    }

    /// Get the data for a specific major index (row in CSR, column in CSC)
    #[inline]
    pub fn get_major(&self, idx: usize) -> (&'a [I], &'a [T]) {
        let start: usize = self.maj_ind[idx].as_();
        let end: usize = self.maj_ind[idx + 1].as_();
        (&self.min_ind[start..end], &self.val[start..end])
    }

    /// Get a specific entry (row, col) from the sparse matrix.
    /// 
    /// This assumes CSR format (row is major index).
    pub fn get_entry(&self, row: usize, col: usize) -> T {
        let (indices, values) = self.get_major(row);
        match indices.binary_search_by(|&i| i.as_().cmp(&col)) {
            Ok(idx) => values[idx],
            Err(_) => T::zero(),
        }
    }
}

/// Extract unique group identifiers from a group assignment vector.
///
/// Returns a sorted vector of unique group IDs, removing duplicates.
pub fn extract_unique_groups(group_ids: &[usize]) -> Vec<usize> {
    let mut unique_groups = group_ids.to_vec();
    unique_groups.sort();
    unique_groups.dedup();
    unique_groups
}

/// Extract indices of cells belonging to each of the two groups.
///
/// Returns a tuple of (group1_indices, group2_indices) where each vector contains
/// the row/column indices of cells belonging to that group.
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

/// Generic version of statistics accumulation that works with any SparseMatrixRef.
/// 
/// Assumes matrix is Genes (rows) x Cells (cols).
pub(crate) fn accumulate_gene_statistics_two_groups_raw<T, N, I>(
    matrix: SparseMatrixRef<T, N, I>,
    group1_indices: &[usize],
    group2_indices: &[usize],
    _n_features: usize,
) -> anyhow::Result<(Vec<T>, Vec<T>, Vec<T>, Vec<T>)>
where
    T: FloatOpsTS,
    N: AsPrimitive<usize> + Send + Sync,
    I: AsPrimitive<usize> + Send + Sync,
{
    let n_genes = matrix.n_rows;
    let n_cells = matrix.n_cols;

    // Create a mapping for group membership to avoid repeated linear searches
    let mut cell_groups = vec![0u8; n_cells];
    for &idx in group1_indices { cell_groups[idx] = 1; }
    for &idx in group2_indices { cell_groups[idx] = 2; }

    // Pre-allocate all accumulation vectors (one per gene/row)
    let mut group1_sums = vec![T::zero(); n_genes];
    let mut group1_sum_squares = vec![T::zero(); n_genes];
    let mut group2_sums = vec![T::zero(); n_genes];
    let mut group2_sum_squares = vec![T::zero(); n_genes];
    
    for row_idx in 0..n_genes {
        let (cols, vals) = matrix.get_major(row_idx);
        for (col_idx, &value) in cols.iter().zip(vals.iter()) {
            let c_idx: usize = col_idx.as_();
            match cell_groups[c_idx] {
                1 => {
                    group1_sums[row_idx] += value;
                    group1_sum_squares[row_idx] += value * value;
                }
                2 => {
                    group2_sums[row_idx] += value;
                    group2_sum_squares[row_idx] += value * value;
                }
                _ => {}
            }
        }
    }

    Ok((group1_sums, group1_sum_squares, group2_sums, group2_sum_squares))
}
