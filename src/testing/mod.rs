//! Statistical testing framework for single-cell data analysis.
//!
//! This module provides a comprehensive suite of statistical tests and methods specifically designed
//! for single-cell RNA-seq data analysis. It includes parametric and non-parametric tests, multiple
//! testing correction methods, and effect size calculations.
//!
//! ## Key Components
//!
//! - **Core Data Structures**: [`TestResult`] and [`MultipleTestResults`] for storing test outcomes
//! - **Test Methods**: [`TestMethod`] enum defining available statistical tests
//! - **Matrix Operations**: [`MatrixStatTests`] trait for running tests on sparse matrices
//!
//! ## Submodules
//!
//! - [`correction`]: Multiple testing correction methods (FDR, Bonferroni, etc.)
//! - [`effect`]: Effect size calculations (Cohen's d, etc.)
//! - [`inference`]: Core statistical inference implementations
//! - [`utils`]: Utility functions for data preparation and validation
//!
//! ## Usage
//!
//! Use the `MatrixStatTests` trait on sparse matrices to perform differential expression
//! analysis with various statistical methods and automatic multiple testing correction.

use single_utilities::traits::FloatOps;
use std::collections::HashMap;

pub mod correction;
pub mod effect;
pub mod inference;

pub mod utils;

/// Statistical test methods available for differential expression analysis.
///
/// This enum defines the different statistical tests that can be applied to single-cell data.
/// Each method has specific assumptions and use cases.
#[derive(Debug, Clone, Copy)]
pub enum TestMethod {
    /// Student's or Welch's t-test for comparing means between two groups.
    /// 
    /// **Use when**: Data is approximately normal, comparing continuous expression values.
    /// **Best for**: Most differential expression analyses in single-cell data.
    TTest(TTestType),
    
    /// Mann-Whitney U test (Wilcoxon rank-sum test) for non-parametric comparison.
    /// 
    /// **Use when**: Data is not normally distributed, or you want a robust rank-based test.
    /// **Best for**: Highly skewed expression data or small sample sizes.
    MannWhitney,
    
    /// Negative binomial test for count data with overdispersion.
    /// 
    /// **Use when**: Working with raw UMI counts and modeling overdispersion.
    /// **Best for**: Count-based differential expression (like DESeq2/edgeR approach).
    NegativeBinomial,
    
    /// Zero-inflated models for data with excess zeros.
    /// 
    /// **Use when**: High proportion of zero values (dropout) needs explicit modeling.
    /// **Best for**: Single-cell data with significant technical dropout.
    ZeroInflated,
}

/// Type of t-test to perform, differing in variance assumptions.
#[derive(Debug, Clone, Copy)]
pub enum TTestType {
    /// Student's t-test assuming equal variances between groups.
    /// 
    /// **Use when**: Groups have similar variance (homoscedasticity).
    /// **Faster** but less robust than Welch's t-test.
    Student,
    
    /// Welch's t-test allowing unequal variances between groups.
    /// 
    /// **Use when**: Groups may have different variances (heteroscedasticity).
    /// **Recommended** for most single-cell analyses as the default choice.
    Welch,
}

/// Alternative hypothesis for statistical tests.
#[derive(Debug, Clone, Copy)]
pub enum Alternative {
    /// Two-sided test: group means are not equal (μ₁ ≠ μ₂).
    /// 
    /// **Most common** choice for differential expression analysis.
    TwoSided,
    
    /// One-sided test: group 1 mean is less than group 2 (μ₁ < μ₂).
    /// 
    /// **Use when**: You specifically want to test for downregulation.
    Less,
    
    /// One-sided test: group 1 mean is greater than group 2 (μ₁ > μ₂).
    /// 
    /// **Use when**: You specifically want to test for upregulation.
    Greater,
}

/// Result of a single statistical test.
///
/// Contains the test statistic, p-value, and optional additional information like effect sizes
/// and confidence intervals. This structure is used for individual gene/feature tests.
///

#[derive(Debug, Clone)]
pub struct TestResult<T> {
    /// The test statistic value (e.g., t-statistic, U statistic)
    pub statistic: T,
    /// The p-value of the test
    pub p_value: T,
    /// Confidence interval for the effect size/difference (if available)
    pub confidence_interval: Option<(T, T)>,
    /// Degrees of freedom (for parametric tests)
    pub degrees_of_freedom: Option<T>,
    /// Effect size measurement (Cohen's d, etc.)
    pub effect_size: Option<T>,
    /// Standard error of the effect size or test statistic
    pub standard_error: Option<T>,
    /// Additional test-specific information
    pub metadata: HashMap<String, T>,
}

impl<T> TestResult<T>
where
    T: FloatOps,
{
    /// Create a new test result with minimal information
    pub fn new(statistic: T, p_value: T) -> Self {
        TestResult {
            statistic,
            p_value,
            confidence_interval: None,
            degrees_of_freedom: None,
            effect_size: None,
            standard_error: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a new test result with effect size
    pub fn with_effect_size(statistic: T, p_value: T, effect_size: T) -> Self {
        TestResult {
            statistic,
            p_value,
            confidence_interval: None,
            degrees_of_freedom: None,
            effect_size: Some(effect_size),
            standard_error: None,
            metadata: HashMap::new(),
        }
    }

    /// Add confidence interval to the result
    pub fn with_confidence_interval(mut self, lower: T, upper: T) -> Self {
        self.confidence_interval = Some((lower, upper));
        self
    }

    /// Add degrees of freedom to the result
    pub fn with_degrees_of_freedom(mut self, df: T) -> Self {
        self.degrees_of_freedom = Some(df);
        self
    }

    /// Add standard error to the result
    pub fn with_standard_error(mut self, se: T) -> Self {
        self.standard_error = Some(se);
        self
    }

    /// Add additional metadata
    pub fn with_metadata(mut self, key: &str, value: T) -> Self {
        self.metadata.insert(key.to_string(), value);
        self
    }

    /// Check if the result is statistically significant at the given threshold
    pub fn is_significant(&self, alpha: T) -> bool {
        self.p_value < alpha
    }
}

/// Results from multiple statistical tests, typically for differential expression analysis.
///
/// This structure contains results from testing multiple features (genes) simultaneously,
/// including multiple testing correction. It's the primary output of differential expression
/// analysis workflows.
///

#[derive(Debug, Clone)]
pub struct MultipleTestResults<T> {
    /// Test statistics for each feature/gene
    pub statistics: Vec<T>,
    /// Raw (unadjusted) p-values
    pub p_values: Vec<T>,
    /// Adjusted p-values (after multiple testing correction)
    pub adjusted_p_values: Option<Vec<T>>,
    /// Effect sizes (if calculated)
    pub effect_sizes: Option<Vec<T>>,
    /// Confidence intervals (if calculated)
    pub confidence_intervals: Option<Vec<(T, T)>>,
    /// Feature-specific metadata
    pub feature_metadata: Option<Vec<HashMap<String, T>>>,
    /// Global metadata about the test
    pub global_metadata: HashMap<String, String>,
}

impl<T> MultipleTestResults<T>
where
    T: FloatOps,
{
    /// Create a new results object from p-values
    pub fn new(statistics: Vec<T>, p_values: Vec<T>) -> Self {
        MultipleTestResults {
            statistics,
            p_values,
            adjusted_p_values: None,
            effect_sizes: None,
            confidence_intervals: None,
            feature_metadata: None,
            global_metadata: HashMap::new(),
        }
    }

    /// Add adjusted p-values to the results
    pub fn with_adjusted_p_values(mut self, adjusted_p_values: Vec<T>) -> Self {
        self.adjusted_p_values = Some(adjusted_p_values);
        self
    }

    /// Add effect sizes to the results
    pub fn with_effect_sizes(mut self, effect_sizes: Vec<T>) -> Self {
        self.effect_sizes = Some(effect_sizes);
        self
    }

    /// Add confidence intervals to the results
    pub fn with_confidence_intervals(mut self, confidence_intervals: Vec<(T, T)>) -> Self {
        self.confidence_intervals = Some(confidence_intervals);
        self
    }

    /// Add global metadata about the test
    pub fn with_global_metadata(mut self, key: &str, value: &str) -> Self {
        self.global_metadata
            .insert(key.to_string(), value.to_string());
        self
    }

    /// Get indices of significant features at the given threshold
    pub fn significant_indices(&self, alpha: T) -> Vec<usize> {
        match &self.adjusted_p_values {
            Some(adj_p_values) => adj_p_values
                .iter()
                .enumerate()
                .filter_map(|(i, &p)| if p < alpha { Some(i) } else { None })
                .collect(),
            None => self
                .p_values
                .iter()
                .enumerate()
                .filter_map(|(i, &p)| if p < alpha { Some(i) } else { None })
                .collect(),
        }
    }

    /// Get the number of significant features at the given threshold
    pub fn num_significant(&self, alpha: T) -> usize {
        self.significant_indices(alpha).len()
    }

    /// Get top n features by p-value
    pub fn top_features(&self, n: usize) -> Vec<usize> {
        let p_values = match &self.adjusted_p_values {
            Some(adj_p) => adj_p,
            None => &self.p_values,
        };

        let mut indices: Vec<usize> = (0..p_values.len()).collect();
        indices.sort_by(|&a, &b| {
            p_values[a]
                .partial_cmp(&p_values[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices.truncate(n);
        indices
    }
}
