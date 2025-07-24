use single_utilities::traits::FloatOps;
use std::collections::HashMap;

pub mod correction;
pub mod effect;
pub mod inference;

pub mod utils;

#[derive(Debug, Clone, Copy)]
pub enum TestMethod {
    TTest(TTestType),
    MannWhitney,
    NegativeBinomial,
    ZeroInflated,
}

#[derive(Debug, Clone, Copy)]
pub enum TTestType {
    Student, // Equal variance
    Welch,   // Unequal variance
}

#[derive(Debug, Clone, Copy)]
pub enum Alternative {
    TwoSided,
    Less,
    Greater,
}

#[derive(Debug, Clone)]
pub struct TestResult<T> {
    /// The test statistic value (e.g., t-statistic, U statistic)
    pub statistic: T,
    /// The p-value of the test
    pub p_value: T,
    /// Confidence interval for the effect size/difference (if available)
    pub confidence_interval: Option<(T, T)>,
    /// Degrees of freedom (for parametric inference)
    pub degrees_of_freedom: Option<T>,
    /// Effect size measurement
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
