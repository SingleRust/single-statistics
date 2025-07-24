// Integration tests for the single_statistics crate
// This file can be expanded with end-to-end tests that verify the library works correctly
// across different modules and in realistic scenarios.

#[cfg(test)]
mod integration_tests {
    use single_statistics::testing::inference::parametric::fast_t_test_from_sums;
    use single_statistics::testing::TTestType;

    #[test]
    fn test_basic_integration() {
        // Basic integration test to ensure the library functions work together
        let sum1 = 50.0;
        let sum_sq1 = 500.0;
        let n1 = 10.0;
        
        let sum2 = 60.0;
        let sum_sq2 = 600.0;
        let n2 = 10.0;
        
        let result = fast_t_test_from_sums(
            sum1, sum_sq1, n1,
            sum2, sum_sq2, n2,
            TTestType::Student
        );
        
        // Basic sanity checks
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert!(result.statistic.is_finite());
    }
}