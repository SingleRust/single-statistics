use crate::testing::utils::{
    extract_unique_groups, get_group_indices, SparseMatrixRef, SprsView,
};
use crate::testing::{
    Alternative, MultipleTestResults, TTestType, TestMethod, TestResult, correction,
};
use single_utilities::traits::FloatOpsTS;
use num_traits::AsPrimitive;

pub mod discrete;
pub mod parametric;
pub mod nonparametric;

/// Statistical testing methods for sparse matrices, particularly suited for single-cell data.
///
/// Implemented for every [`sprs`] matrix and for [`SparseMatrixRef`], so the same calls
/// work over an owned `CsMat`, a borrowed view, or raw slices handed across an FFI
/// boundary such as PyO3.
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

    /// Perform Wilcoxon signed-rank tests over paired cells for all genes.
    ///
    /// `group1_indices[i]` is paired with `group2_indices[i]`; the slices must be
    /// the same length and ordered consistently.
    fn wilcoxon_signed_rank_test(
        &self,
        group1_indices: &[usize],
        group2_indices: &[usize],
        alternative: Alternative,
    ) -> anyhow::Result<Vec<TestResult<f64>>>;

    /// Perform Kruskal-Wallis H tests across two or more groups for all genes.
    ///
    /// `group_ids` assigns every cell (column) to a group.
    fn kruskal_wallis_test(&self, group_ids: &[usize]) -> anyhow::Result<Vec<TestResult<f64>>>;

    /// Comprehensive differential expression analysis with multiple testing correction.
    fn differential_expression(
        &self,
        group_ids: &[usize],
        test_method: TestMethod,
    ) -> anyhow::Result<MultipleTestResults<f64>>;
}

/// Blanket implementation for every [`sprs`] sparse matrix — owned `CsMat`, borrowed
/// `CsMatView`, outer-sliced views, and any other storage that derefs to slices.
///
/// See [`SprsView`] for the orientation contract (major axis = features) and for how
/// outer-sliced views are rebased.
impl<T, I, Iptr, IptrStorage, IndStorage, DataStorage> MatrixStatTests<T>
    for sprs::CsMatBase<T, I, IptrStorage, IndStorage, DataStorage, Iptr>
where
    T: FloatOpsTS,
    I: sprs::indexing::SpIndex + AsPrimitive<usize>,
    Iptr: sprs::indexing::SpIndex + AsPrimitive<usize>,
    IptrStorage: std::ops::Deref<Target = [Iptr]> + Send + Sync,
    IndStorage: std::ops::Deref<Target = [I]> + Send + Sync,
    DataStorage: std::ops::Deref<Target = [T]> + Send + Sync,
    f64: std::convert::From<T>,
{
    fn t_test(
        &self,
        group1_indices: &[usize],
        group2_indices: &[usize],
        test_type: TTestType,
    ) -> anyhow::Result<Vec<TestResult<f64>>> {
        parametric::t_test_sparse(
            SprsView::new(self).as_matrix_ref(),
            group1_indices,
            group2_indices,
            test_type,
        )
    }

    fn mann_whitney_test(
        &self,
        group1_indices: &[usize],
        group2_indices: &[usize],
        alternative: Alternative,
    ) -> anyhow::Result<Vec<TestResult<f64>>> {
        nonparametric::mann_whitney_sparse(
            SprsView::new(self).as_matrix_ref(),
            group1_indices,
            group2_indices,
            alternative,
        )
    }

    fn fisher_exact_test(
        &self,
        group1_indices: &[usize],
        group2_indices: &[usize],
        alternative: Alternative,
    ) -> anyhow::Result<Vec<TestResult<T>>> {
        discrete::fisher_exact_sparse(
            SprsView::new(self).as_matrix_ref(),
            group1_indices,
            group2_indices,
            alternative,
        )
    }

    fn wilcoxon_signed_rank_test(
        &self,
        group1_indices: &[usize],
        group2_indices: &[usize],
        alternative: Alternative,
    ) -> anyhow::Result<Vec<TestResult<f64>>> {
        nonparametric::wilcoxon_signed_rank_sparse(
            SprsView::new(self).as_matrix_ref(),
            group1_indices,
            group2_indices,
            alternative,
        )
    }

    fn kruskal_wallis_test(&self, group_ids: &[usize]) -> anyhow::Result<Vec<TestResult<f64>>> {
        nonparametric::kruskal_wallis_sparse(SprsView::new(self).as_matrix_ref(), group_ids)
    }

    fn differential_expression(
        &self,
        group_ids: &[usize],
        test_method: TestMethod,
    ) -> anyhow::Result<MultipleTestResults<f64>> {
        SprsView::new(self).as_matrix_ref().differential_expression(group_ids, test_method)
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

    fn wilcoxon_signed_rank_test(
        &self,
        group1_indices: &[usize],
        group2_indices: &[usize],
        alternative: Alternative,
    ) -> anyhow::Result<Vec<TestResult<f64>>> {
        nonparametric::wilcoxon_signed_rank_sparse(
            *self,
            group1_indices,
            group2_indices,
            alternative,
        )
    }

    fn kruskal_wallis_test(&self, group_ids: &[usize]) -> anyhow::Result<Vec<TestResult<f64>>> {
        nonparametric::kruskal_wallis_sparse(*self, group_ids)
    }

    fn differential_expression(
        &self,
        group_ids: &[usize],
        test_method: TestMethod,
    ) -> anyhow::Result<MultipleTestResults<f64>> {
        // Kruskal-Wallis is the one method here that generalises beyond two groups,
        // so it is dispatched before the two-group gate below.
        if let TestMethod::KruskalWallis = test_method {
            return summarize(&self.kruskal_wallis_test(group_ids)?, "kruskal_wallis");
        }

        let unique_groups = extract_unique_groups(group_ids);
        if unique_groups.len() != 2 {
            return Err(anyhow::anyhow!(
                "Currently only two-group comparisons are supported for this test method (use TestMethod::KruskalWallis for {} groups)",
                unique_groups.len()
            ));
        }

        let (g1, g2) = get_group_indices(group_ids, &unique_groups);
        let two_sided = Alternative::TwoSided;

        match test_method {
            TestMethod::TTest(t) => summarize(&self.t_test(&g1, &g2, t)?, "t_test"),
            TestMethod::MannWhitney => {
                summarize(&self.mann_whitney_test(&g1, &g2, two_sided)?, "mann_whitney")
            }
            TestMethod::WilcoxonSignedRank => summarize(
                &self.wilcoxon_signed_rank_test(&g1, &g2, two_sided)?,
                "wilcoxon_signed_rank",
            ),
            TestMethod::FisherExact => {
                summarize(&self.fisher_exact_test(&g1, &g2, two_sided)?, "fisher_exact")
            }
            TestMethod::NegativeBinomial => summarize(
                &discrete::negative_binomial_sparse(*self, &g1, &g2, two_sided)?,
                "negative_binomial",
            ),
            TestMethod::ZeroInflated => summarize(
                &discrete::zero_inflated_sparse(*self, &g1, &g2, two_sided)?,
                "zero_inflated",
            ),
            // Dispatched above, before the two-group gate.
            TestMethod::KruskalWallis => unreachable!("handled before the two-group gate"),
        }
    }
}

/// Fold per-gene results into the multi-test summary: BH correction plus whatever
/// effect sizes, intervals and metadata the individual tests produced.
fn summarize<T>(
    results: &[TestResult<T>],
    label: &'static str,
) -> anyhow::Result<MultipleTestResults<f64>>
where
    T: single_utilities::traits::FloatOps,
{
    let statistics: Vec<f64> = results
        .iter()
        .map(|r| r.statistic.to_f64().unwrap_or(f64::NAN))
        .collect();
    let p_values: Vec<f64> = results
        .iter()
        .map(|r| r.p_value.to_f64().unwrap_or(1.0))
        .collect();
    let adjusted = correction::benjamini_hochberg_correction(&p_values)?;

    let mut out = MultipleTestResults::new(statistics, p_values)
        .with_adjusted_p_values(adjusted)
        .with_global_metadata("test_type", label);

    if results.iter().any(|r| r.effect_size.is_some()) {
        out = out.with_effect_sizes(
            results
                .iter()
                .map(|r| r.effect_size.and_then(|e| e.to_f64()).unwrap_or(0.0))
                .collect(),
        );
    }

    if results.iter().any(|r| r.confidence_interval.is_some()) {
        out = out.with_confidence_intervals(
            results
                .iter()
                .map(|r| match r.confidence_interval {
                    Some((lo, hi)) => (
                        lo.to_f64().unwrap_or(f64::NAN),
                        hi.to_f64().unwrap_or(f64::NAN),
                    ),
                    None => (f64::NAN, f64::NAN),
                })
                .collect(),
        );
    }

    out.feature_metadata = Some(
        results
            .iter()
            .map(|r| {
                r.metadata
                    .iter()
                    .map(|(k, v)| (*k, v.to_f64().unwrap_or(f64::NAN)))
                    .collect()
            })
            .collect(),
    );

    Ok(out)
}
