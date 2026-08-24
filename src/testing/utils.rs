//! Utility functions for statistical testing operations.

use single_utilities::traits::FloatOpsTS;
use num_traits::AsPrimitive;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

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

/// Borrowed view of an [`sprs`] matrix, ready to be handed to this crate's
/// statistical routines as a [`SparseMatrixRef`].
///
/// # Why this type exists
///
/// An outer-sliced sprs view (from [`sprs::CsMatBase::slice_outer`] and friends)
/// narrows its `indptr` while keeping the *original absolute* offsets, but slices
/// `indices`/`data` alongside. The three arrays therefore cannot be read as a flat
/// CSR triple until the offsets are rebased, and rebasing needs somewhere to put the
/// result. This type owns that buffer when it is needed and borrows when it is not,
/// so the common (unsliced) case stays entirely zero-copy.
///
/// # Orientation
///
/// The view is *major-oriented*: every statistical routine in this crate treats the
/// major axis as **features (genes)** and the minor axis as **observations (cells)**.
/// Both of these map directly:
///
/// - a **CSR** matrix of shape `genes x cells`
/// - a **CSC** matrix of shape `cells x genes`
///
/// A CSR `cells x genes` matrix would be read with cells as features; transpose it or
/// convert its storage order first.
///
/// ```no_run
/// # use sprs::{CsMat, TriMat};
/// # use single_statistics::testing::utils::SprsView;
/// # let matrix: CsMat<f64> = TriMat::new((2, 2)).to_csr();
/// let view = SprsView::new(&matrix);
/// let matrix_ref = view.as_matrix_ref();
/// ```
pub struct SprsView<'a, T, I, Iptr>
where
    Iptr: Clone,
{
    indptr: std::borrow::Cow<'a, [Iptr]>,
    indices: &'a [I],
    data: &'a [T],
    n_major: usize,
    n_minor: usize,
}

impl<'a, T, I, Iptr> SprsView<'a, T, I, Iptr>
where
    I: sprs::indexing::SpIndex,
    Iptr: sprs::indexing::SpIndex,
{
    /// Wrap an sprs matrix of any storage order or slicing state.
    ///
    /// Borrows throughout unless the matrix is outer-sliced, in which case only the
    /// `indptr` is copied and rebased.
    pub fn new<IptrStorage, IndStorage, DataStorage>(
        matrix: &'a sprs::CsMatBase<T, I, IptrStorage, IndStorage, DataStorage, Iptr>,
    ) -> Self
    where
        IptrStorage: std::ops::Deref<Target = [Iptr]>,
        IndStorage: std::ops::Deref<Target = [I]>,
        DataStorage: std::ops::Deref<Target = [T]>,
    {
        // `proper_indptr` yields Cow::Borrowed when the matrix is already proper,
        // and Cow::Owned with the offsets rebased to zero when it is not.
        let indptr = matrix.proper_indptr();

        // sprs puts the major axis first regardless of CSR/CSC, so the lane count is
        // rows for CSR and cols for CSC.
        let (n_major, n_minor) = if matrix.is_csr() {
            (matrix.rows(), matrix.cols())
        } else {
            (matrix.cols(), matrix.rows())
        };

        Self {
            indptr,
            indices: matrix.indices(),
            data: matrix.data(),
            n_major,
            n_minor,
        }
    }

    /// Borrow this view as the container-agnostic [`SparseMatrixRef`].
    pub fn as_matrix_ref(&self) -> SparseMatrixRef<'_, T, Iptr, I> {
        SparseMatrixRef {
            maj_ind: &self.indptr,
            min_ind: self.indices,
            val: self.data,
            n_rows: self.n_major,
            n_cols: self.n_minor,
        }
    }

    /// Number of features (major lanes).
    pub fn n_features(&self) -> usize {
        self.n_major
    }

    /// Number of observations (minor slots).
    pub fn n_observations(&self) -> usize {
        self.n_minor
    }

    /// Whether the wrapper had to copy and rebase the `indptr`, which happens only
    /// for outer-sliced views.
    pub fn is_rebased(&self) -> bool {
        matches!(self.indptr, std::borrow::Cow::Owned(_))
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

/// Per-gene summary statistics for a two-group comparison: the sums and sums of
/// squares of group 1 followed by those of group 2, each indexed by gene.
pub(crate) type TwoGroupSums<T> = (Vec<T>, Vec<T>, Vec<T>, Vec<T>);

/// Generic version of statistics accumulation that works with any SparseMatrixRef.
///
/// Assumes matrix is Genes (rows) x Cells (cols).
///
/// Rows are independent — each one owns its output slot — so the traversal runs in
/// parallel. Accumulating into locals rather than indexing four heap vectors also
/// keeps the inner loop in registers.
pub(crate) fn accumulate_gene_statistics_two_groups_raw<T, N, I>(
    matrix: SparseMatrixRef<T, N, I>,
    group1_indices: &[usize],
    group2_indices: &[usize],
) -> anyhow::Result<TwoGroupSums<T>>
where
    T: FloatOpsTS,
    N: AsPrimitive<usize> + Send + Sync,
    I: AsPrimitive<usize> + Send + Sync,
{
    let n_genes = matrix.n_rows;
    let n_cells = matrix.n_cols;

    // Create a mapping for group membership to avoid repeated linear searches
    let mut cell_groups = vec![0u8; n_cells];
    for &idx in group1_indices {
        if idx < n_cells { cell_groups[idx] = 1; }
    }
    for &idx in group2_indices {
        if idx < n_cells { cell_groups[idx] = 2; }
    }

    let per_gene: Vec<(T, T, T, T)> = (0..n_genes)
        .into_par_iter()
        .map(|row_idx| {
            let (cols, vals) = matrix.get_major(row_idx);
            let mut g1_sum = T::zero();
            let mut g1_sq = T::zero();
            let mut g2_sum = T::zero();
            let mut g2_sq = T::zero();

            for (col_idx, &value) in cols.iter().zip(vals.iter()) {
                let c_idx: usize = col_idx.as_();
                match cell_groups[c_idx] {
                    1 => {
                        g1_sum += value;
                        g1_sq += value * value;
                    }
                    2 => {
                        g2_sum += value;
                        g2_sq += value * value;
                    }
                    _ => {}
                }
            }
            (g1_sum, g1_sq, g2_sum, g2_sq)
        })
        .collect();

    let mut group1_sums = Vec::with_capacity(n_genes);
    let mut group1_sum_squares = Vec::with_capacity(n_genes);
    let mut group2_sums = Vec::with_capacity(n_genes);
    let mut group2_sum_squares = Vec::with_capacity(n_genes);
    for (g1_sum, g1_sq, g2_sum, g2_sq) in per_gene {
        group1_sums.push(g1_sum);
        group1_sum_squares.push(g1_sq);
        group2_sums.push(g2_sum);
        group2_sum_squares.push(g2_sq);
    }

    Ok((group1_sums, group1_sum_squares, group2_sums, group2_sum_squares))
}

/// Small deterministic PRNG (xorshift64). Keeps permutation tests reproducible
/// without pulling in `rand`.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed | 1) // xorshift stalls at zero
    }

    pub(crate) fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    pub(crate) fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    pub(crate) fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i + 1));
        }
    }
}

/// The standard normal, built once. These lookups sit inside per-gene parallel maps.
pub(crate) fn standard_normal() -> &'static statrs::distribution::Normal {
    static N: std::sync::LazyLock<statrs::distribution::Normal> = std::sync::LazyLock::new(|| {
        statrs::distribution::Normal::new(0.0, 1.0).expect("standard normal is valid")
    });
    &N
}
