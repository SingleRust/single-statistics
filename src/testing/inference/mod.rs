use crate::testing::utils::{extract_unique_groups, get_group_indices};
use crate::testing::{
    Alternative, MultipleTestResults, TTestType, TestMethod, TestResult, correction,
};
use nalgebra_sparse::CsrMatrix;
use single_utilities::traits::FloatOpsTS;

pub mod discrete;

pub mod parametric;

pub mod nonparametric;

/// Statistical testing methods for sparse matrices, particularly suited for single-cell data.
///
/// This trait extends sparse matrix types (like `CsrMatrix`) with statistical testing capabilities.
/// It provides methods for differential expression analysis and other statistical comparisons
/// optimized for single-cell RNA-seq data.
///
/// # Matrix Format
///
/// The expected matrix format is **genes × cells** (features × observations), where:
/// - Rows represent genes/features
/// - Columns represent cells/observations
/// - Values represent expression levels (normalized counts, log-transformed, etc.)
///

pub trait MatrixStatTests<T>
where
    T: FloatOpsTS,
{
    /// Perform t-tests comparing two groups of cells for all genes.
    ///
    /// Runs Student's or Welch's t-test for each gene (row) in the matrix, comparing
    /// expression levels between the specified groups of cells (columns).
    ///
    /// # Arguments
    ///
    /// * `group1_indices` - Column indices for the first group of cells
    /// * `group2_indices` - Column indices for the second group of cells  
    /// * `test_type` - Type of t-test (`Student` or `Welch`)
    ///
    /// # Returns
    ///
    /// Vector of `TestResult` objects, one per gene, containing test statistics and p-values.
    ///

    fn t_test(
        &self,
        group1_indices: &[usize],
        group2_indices: &[usize],
        test_type: TTestType,
    ) -> anyhow::Result<Vec<TestResult<f64>>>;

    /// Perform Mann-Whitney U tests comparing two groups of cells for all genes.
    ///
    /// Runs non-parametric Mann-Whitney U (Wilcoxon rank-sum) tests for each gene,
    /// comparing the distributions between the specified groups.
    ///
    /// # Arguments
    ///
    /// * `group1_indices` - Column indices for the first group of cells
    /// * `group2_indices` - Column indices for the second group of cells
    /// * `alternative` - Type of alternative hypothesis (two-sided, less, greater)
    ///
    /// # Returns
    ///
    /// Vector of `TestResult` objects containing U statistics and p-values.
    fn mann_whitney_test(
        &self,
        group1_indices: &[usize],
        group2_indices: &[usize],
        alternative: Alternative,
    ) -> anyhow::Result<Vec<TestResult<f64>>>;

    /// Comprehensive differential expression analysis with multiple testing correction.
    ///
    /// This is the main method for differential expression analysis. It performs the
    /// specified statistical test on all genes and applies multiple testing correction
    /// to control the false discovery rate.
    ///
    /// # Arguments
    ///
    /// * `group_ids` - Vector assigning each cell to a group (currently supports 2 groups)
    /// * `test_method` - Statistical test to perform
    ///
    /// # Returns
    ///
    /// `MultipleTestResults` containing statistics, p-values, adjusted p-values, and
    /// metadata for all genes.
    ///

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
                // Run t-tests
                let results = self.t_test(&group1_indices, &group2_indices, test_type)?;

                // Extract statistics and p-values
                let statistics: Vec<_> = results.iter().map(|r| r.statistic).collect();
                let p_values: Vec<_> = results.iter().map(|r| r.p_value).collect();

                // Apply multiple testing correction
                let adjusted_p_values = correction::benjamini_hochberg_correction(&p_values)?;

                // Extract effect sizes if available
                let effect_sizes = results
                    .iter()
                    .map(|r| r.effect_size)
                    .collect::<Option<Vec<_>>>()
                    .unwrap_or_else(|| vec![0.0; results.len()])
                    .into_iter()
                    .filter_map(|x| Option::from(x))
                    .collect::<Vec<_>>();

                let mut result = MultipleTestResults::new(statistics, p_values)
                    .with_adjusted_p_values(adjusted_p_values)
                    .with_global_metadata("test_type", "t_test");

                if !effect_sizes.is_empty() {
                    result = result.with_effect_sizes(effect_sizes);
                }

                Ok(result)
            }

            TestMethod::MannWhitney => {
                // Run Mann-Whitney tests
                let results = self.mann_whitney_test(
                    &group1_indices,
                    &group2_indices,
                    Alternative::TwoSided,
                )?;

                // Extract statistics and p-values
                let statistics: Vec<_> = results.iter().map(|r| r.statistic).collect();
                let p_values: Vec<_> = results.iter().map(|r| r.p_value).collect();

                // Apply multiple testing correction
                let adjusted_p_values = correction::benjamini_hochberg_correction(&p_values)?;

                Ok(MultipleTestResults::new(statistics, p_values)
                    .with_adjusted_p_values(adjusted_p_values)
                    .with_global_metadata("test_type", "mann_whitney"))
            }

            // Implement other test methods similarly
            _ => Err(anyhow::anyhow!("Test method not implemented yet")),
        }
    }
}
