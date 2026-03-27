use crate::testing::utils::{extract_unique_groups, get_group_indices, SparseMatrixRef};
use crate::testing::{
    Alternative, MultipleTestResults, TTestType, TestMethod, TestResult, correction,
};
use nalgebra_sparse::CsrMatrix;
use single_utilities::traits::FloatOpsTS;
use num_traits::AsPrimitive;

pub mod discrete;
pub mod parametric;
pub mod nonparametric;

/// Statistical testing methods for sparse matrices, particularly suited for single-cell data.
///
/// This trait extends sparse matrix types (like `CsrMatrix` or `SparseMatrixRef`) with 
/// statistical testing capabilities.
pub trait MatrixStatTests<T>
where
    T: FloatOpsTS,
{
    /// Perform t-tests comparing two groups of cells for all genes.
    fn t_test(
        &self,
        group1_indices: &[usize],
        group2_indices: &[usize],
        test_type: TTestType,
    ) -> anyhow::Result<Vec<TestResult<f64>>>;

    /// Perform Mann-Whitney U tests comparing two groups of cells for all genes.
    fn mann_whitney_test(
        &self,
        group1_indices: &[usize],
        group2_indices: &[usize],
        alternative: Alternative,
    ) -> anyhow::Result<Vec<TestResult<f64>>>;

    /// Perform Fisher's Exact tests comparing expression frequency between two groups.
    fn fisher_exact_test(
        &self,
        group1_indices: &[usize],
        group2_indices: &[usize],
        alternative: Alternative,
    ) -> anyhow::Result<Vec<TestResult<T>>>;

    /// Comprehensive differential expression analysis with multiple testing correction.
    fn differential_expression(
        &self,
        group_ids: &[usize],
        test_method: TestMethod,
    ) -> anyhow::Result<MultipleTestResults<f64>>;
}

impl<T> MatrixStatTests<T> for CsrMatrix<T>
where
    T: FloatOpsTS,
    f64: std::convert::From<T>,
{
    fn t_test(
        &self,
        group1_indices: &[usize],
        group2_indices: &[usize],
        test_type: TTestType,
    ) -> anyhow::Result<Vec<TestResult<f64>>> {
        parametric::t_test_matrix_groups(self, group1_indices, group2_indices, test_type)
    }

    fn mann_whitney_test(
        &self,
        group1_indices: &[usize],
        group2_indices: &[usize],
        alternative: Alternative,
    ) -> anyhow::Result<Vec<TestResult<f64>>> {
        nonparametric::mann_whitney_matrix_groups(self, group1_indices, group2_indices, alternative)
    }

    fn fisher_exact_test(
        &self,
        group1_indices: &[usize],
        group2_indices: &[usize],
        alternative: Alternative,
    ) -> anyhow::Result<Vec<TestResult<T>>> {
        let smr = SparseMatrixRef {
            maj_ind: self.row_offsets(),
            min_ind: self.col_indices(),
            val: self.values(),
            n_rows: self.nrows(),
            n_cols: self.ncols(),
        };
        discrete::fisher_exact_sparse(smr, group1_indices, group2_indices, alternative)
    }

    fn differential_expression(
        &self,
        group_ids: &[usize],
        test_method: TestMethod,
    ) -> anyhow::Result<MultipleTestResults<f64>> {
        let smr = SparseMatrixRef {
            maj_ind: self.row_offsets(),
            min_ind: self.col_indices(),
            val: self.values(),
            n_rows: self.nrows(),
            n_cols: self.ncols(),
        };
        smr.differential_expression(group_ids, test_method)
    }
}

impl<'a, T, N, I> MatrixStatTests<T> for SparseMatrixRef<'a, T, N, I>
where
    T: FloatOpsTS,
    N: AsPrimitive<usize> + Send + Sync,
    I: AsPrimitive<usize> + Send + Sync,
    f64: std::convert::From<T>,
{
    fn t_test(
        &self,
        group1_indices: &[usize],
        group2_indices: &[usize],
        test_type: TTestType,
    ) -> anyhow::Result<Vec<TestResult<f64>>> {
        parametric::t_test_sparse(*self, group1_indices, group2_indices, test_type)
    }

    fn mann_whitney_test(
        &self,
        group1_indices: &[usize],
        group2_indices: &[usize],
        alternative: Alternative,
    ) -> anyhow::Result<Vec<TestResult<f64>>> {
        nonparametric::mann_whitney_sparse(*self, group1_indices, group2_indices, alternative)
    }

    fn fisher_exact_test(
        &self,
        group1_indices: &[usize],
        group2_indices: &[usize],
        alternative: Alternative,
    ) -> anyhow::Result<Vec<TestResult<T>>> {
        discrete::fisher_exact_sparse(*self, group1_indices, group2_indices, alternative)
    }

    fn differential_expression(
        &self,
        group_ids: &[usize],
        test_method: TestMethod,
    ) -> anyhow::Result<MultipleTestResults<f64>> {
        let unique_groups = extract_unique_groups(group_ids);
        if unique_groups.len() != 2 {
            return Err(anyhow::anyhow!(
                "Currently only two-group comparisons are supported"
            ));
        }

        let (group1_indices, group2_indices) = get_group_indices(group_ids, &unique_groups);

        match test_method {
            TestMethod::TTest(test_type) => {
                let results = self.t_test(&group1_indices, &group2_indices, test_type)?;
                let statistics: Vec<_> = results.iter().map(|r| r.statistic).collect();
                let p_values: Vec<_> = results.iter().map(|r| r.p_value).collect();
                let adjusted_p_values = correction::benjamini_hochberg_correction(&p_values)?;

                let effect_sizes: Vec<f64> = results
                    .iter()
                    .map(|r| r.effect_size.unwrap_or(0.0))
                    .collect();

                Ok(MultipleTestResults::new(statistics, p_values)
                    .with_adjusted_p_values(adjusted_p_values)
                    .with_effect_sizes(effect_sizes)
                    .with_global_metadata("test_type", "t_test"))
            }

            TestMethod::MannWhitney => {
                let results = self.mann_whitney_test(
                    &group1_indices,
                    &group2_indices,
                    Alternative::TwoSided,
                )?;
                let statistics: Vec<_> = results.iter().map(|r| r.statistic).collect();
                let p_values: Vec<_> = results.iter().map(|r| r.p_value).collect();
                let adjusted_p_values = correction::benjamini_hochberg_correction(&p_values)?;

                Ok(MultipleTestResults::new(statistics, p_values)
                    .with_adjusted_p_values(adjusted_p_values)
                    .with_global_metadata("test_type", "mann_whitney"))
            }

            TestMethod::FisherExact => {
                let results = self.fisher_exact_test(
                    &group1_indices,
                    &group2_indices,
                    Alternative::TwoSided,
                )?;
                let statistics: Vec<_> = results.iter().map(|r| r.statistic.to_f64().unwrap()).collect();
                let p_values: Vec<_> = results.iter().map(|r| r.p_value.to_f64().unwrap()).collect();
                let adjusted_p_values = correction::benjamini_hochberg_correction(&p_values)?;

                Ok(MultipleTestResults::new(statistics, p_values)
                    .with_adjusted_p_values(adjusted_p_values)
                    .with_global_metadata("test_type", "fisher_exact"))
            }
            _ => Err(anyhow::anyhow!("Test method not implemented yet")),
        }
    }
}
