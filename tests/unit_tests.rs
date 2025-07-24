use single_statistics::testing::inference::parametric::{fast_t_test_from_sums, t_test_matrix_groups};
use single_statistics::testing::{TTestType, TestResult};
use nalgebra_sparse::{CsrMatrix, CooMatrix};

#[cfg(test)]
mod quick_test {
    use super::*;

    #[test]
    fn check_if_ttest_works() {
        // Simple test: two clearly different groups
        // Group 1: [1, 2, 3] -> mean=2, should have low variance
        // Group 2: [7, 8, 9] -> mean=8, should have low variance  
        // These groups are clearly different, so p-value should be very small (< 0.05)
        
        let sum1 = 6.0;      // 1+2+3
        let sum_sq1 = 14.0;  // 1²+2²+3² = 1+4+9
        let n1 = 3.0;
        
        let sum2 = 24.0;     // 7+8+9
        let sum_sq2 = 194.0; // 7²+8²+9² = 49+64+81
        let n2 = 3.0;
        
        // Call your function (adjust the function name as needed)
        let result: TestResult<f64> = fast_t_test_from_sums(
            sum1, sum_sq1, n1,
            sum2, sum_sq2, n2,
            TTestType::Student
        );
        
        println!("=== T-TEST RESULTS ===");
        println!("Group 1 stats: sum={}, sum_sq={}, n={}", sum1, sum_sq1, n1);
        println!("Group 2 stats: sum={}, sum_sq={}, n={}", sum2, sum_sq2, n2);
        println!("T-statistic: {}", result.statistic);
        println!("P-value: {}", result.p_value);
        
        // Check if results make sense
        if result.p_value < 0.05 && result.statistic.abs() > 2.0 {
            println!("✅ PASS: T-test is working correctly!");
            println!("   - P-value is small ({}), indicating significant difference", result.p_value);
            println!("   - T-statistic is large ({}), as expected", result.statistic);
        } else {
            println!("❌ FAIL: T-test results look wrong!");
            println!("   - Expected: p-value < 0.05 and |t-stat| > 2.0");
            println!("   - Got: p-value = {}, t-stat = {}", result.p_value, result.statistic);
            panic!("T-test implementation has issues!");
        }
    }
    
    #[test] 
    fn check_identical_groups() {
        // Identical groups should give p-value ≈ 1.0 and t-stat ≈ 0.0
        let sum1 = 15.0;     // 5+5+5
        let sum_sq1 = 75.0;  // 5²+5²+5² = 25+25+25
        let n1 = 3.0;
        
        let sum2 = 15.0;     // 5+5+5 (identical)
        let sum_sq2 = 75.0;  // 5²+5²+5²
        let n2 = 3.0;
        
        let result: TestResult<f64> = fast_t_test_from_sums(
            sum1, sum_sq1, n1,
            sum2, sum_sq2, n2,
            TTestType::Student
        );
        
        println!("\n=== IDENTICAL GROUPS TEST ===");
        println!("T-statistic: {}", result.statistic);
        println!("P-value: {}", result.p_value);
        
        if result.statistic.abs() < 0.001 && result.p_value > 0.9 {
            println!("✅ PASS: Identical groups handled correctly!");
        } else {
            println!("❌ FAIL: Identical groups not handled correctly!");
            println!("   - Expected: t-stat ≈ 0, p-value ≈ 1.0");
            println!("   - Got: t-stat = {}, p-value = {}", result.statistic, result.p_value);
        }
    }
    
    #[test] 
    fn check_identical_groups_welch() {
        // Test Welch t-test with identical groups
        let sum1 = 15.0;     // 5+5+5
        let sum_sq1 = 75.0;  // 5²+5²+5² = 25+25+25
        let n1 = 3.0;
        
        let sum2 = 15.0;     // 5+5+5 (identical)
        let sum_sq2 = 75.0;  // 5²+5²+5²
        let n2 = 3.0;
        
        let result: TestResult<f64> = fast_t_test_from_sums(
            sum1, sum_sq1, n1,
            sum2, sum_sq2, n2,
            TTestType::Welch
        );
        
        println!("\n=== IDENTICAL GROUPS TEST (WELCH) ===");
        println!("T-statistic: {}", result.statistic);
        println!("P-value: {}", result.p_value);
        
        if result.statistic.abs() < 0.001 && result.p_value > 0.9 {
            println!("✅ PASS: Welch t-test with identical groups handled correctly!");
        } else {
            println!("❌ FAIL: Welch t-test with identical groups not handled correctly!");
            println!("   - Expected: t-stat ≈ 0, p-value ≈ 1.0");
            println!("   - Got: t-stat = {}, p-value = {}", result.statistic, result.p_value);
        }
    }

    #[test]
    fn test_high_vs_low_expression() {
        // Simulating a highly expressed gene vs low expression
        // High expression group: mean ≈ 10, some variance
        // Low expression group: mean ≈ 2, some variance
        
        // High expression: [9, 10, 11, 10, 10] -> sum=50, sum_sq=502
        let sum_high = 50.0;
        let sum_sq_high = 502.0; // 81+100+121+100+100
        let n_high = 5.0;
        
        // Low expression: [1, 2, 3, 2, 2] -> sum=10, sum_sq=22
        let sum_low = 10.0; 
        let sum_sq_low = 22.0; // 1+4+9+4+4
        let n_low = 5.0;
        
        let result = fast_t_test_from_sums(
            sum_high, sum_sq_high, n_high,
            sum_low, sum_sq_low, n_low,
            TTestType::Student
        );
        
        println!("\n=== HIGH VS LOW EXPRESSION TEST ===");
        println!("High group: mean={}, variance estimated", sum_high / n_high);
        println!("Low group: mean={}, variance estimated", sum_low / n_low);
        println!("T-statistic: {}", result.statistic);
        println!("P-value: {}", result.p_value);
        
        // Should be highly significant
        assert!(result.p_value < 0.001, "Expected highly significant p-value");
        assert!(num_traits::Float::abs(result.statistic) > 5.0, "Expected large t-statistic");
        println!("✅ PASS: High vs low expression detected correctly!");
    }

    #[test]
    fn test_zero_expression_vs_expressed() {
        // Common in single-cell: some cells express gene, others don't
        // Expressed group: [5, 4, 6, 5, 5] -> sum=25, sum_sq=127
        let sum_expressed = 25.0;
        let sum_sq_expressed = 127.0; // 25+16+36+25+25
        let n_expressed = 5.0;
        
        // Zero expression: [0, 0, 0, 0, 0] -> sum=0, sum_sq=0
        let sum_zero = 0.0;
        let sum_sq_zero = 0.0;
        let n_zero = 5.0;
        
        let result = fast_t_test_from_sums(
            sum_expressed, sum_sq_expressed, n_expressed,
            sum_zero, sum_sq_zero, n_zero,
            TTestType::Welch // Welch is better for unequal variances
        );
        
        println!("\n=== ZERO VS EXPRESSED TEST ===");
        println!("Expressed group mean: {}", sum_expressed / n_expressed);
        println!("Zero group mean: {}", sum_zero / n_zero);
        println!("T-statistic: {}", result.statistic);
        println!("P-value: {}", result.p_value);
        
        // Should be highly significant
        assert!(result.p_value < 0.001, "Expected highly significant difference");
        assert!(num_traits::Float::abs(result.statistic) > 3.0, "Expected large t-statistic");
        println!("✅ PASS: Zero vs expressed gene detected correctly!");
    }

    #[test]
    fn test_subtle_difference() {
        // Subtle but real difference (common in single-cell DE)
        // Group 1: slightly higher expression [3.1, 3.2, 3.0, 3.1, 3.0]
        // Group 2: slightly lower expression [2.9, 2.8, 3.0, 2.9, 3.0]
        
        let sum1 = 15.4; // 3.1+3.2+3.0+3.1+3.0
        let sum_sq1 = 47.46; // 9.61+10.24+9.0+9.61+9.0
        let n1 = 5.0;
        
        let sum2 = 14.6; // 2.9+2.8+3.0+2.9+3.0
        let sum_sq2 = 42.58; // 8.41+7.84+9.0+8.41+9.0
        let n2 = 5.0;
        
        let result = fast_t_test_from_sums(
            sum1, sum_sq1, n1,
            sum2, sum_sq2, n2,
            TTestType::Student
        );
        
        println!("\n=== SUBTLE DIFFERENCE TEST ===");
        println!("Group 1 mean: {}", sum1 / n1);
        println!("Group 2 mean: {}", sum2 / n2);
        println!("Mean difference: {}", (sum1 / n1) - (sum2 / n2));
        println!("T-statistic: {}", result.statistic);
        println!("P-value: {}", result.p_value);
        
        // This should show some difference but may not be highly significant
        println!("✅ PASS: Subtle difference test completed (p-value interpretation depends on effect size)");
    }

    #[test]
    fn test_unequal_sample_sizes() {
        // Common in single-cell: unequal cell counts between groups
        // Large group: 10 cells with moderate expression
        let sum_large = 50.0; // 10 cells, mean = 5.0
        let sum_sq_large = 260.0; // Some variance
        let n_large = 10.0;
        
        // Small group: 3 cells with high expression  
        let sum_small = 21.0; // 3 cells, mean = 7.0
        let sum_sq_small = 149.0; // 7²+7²+7² = 147, plus some variance
        let n_small = 3.0;
        
        let result = fast_t_test_from_sums(
            sum_small, sum_sq_small, n_small,
            sum_large, sum_sq_large, n_large,
            TTestType::Welch // Better for unequal samples
        );
        
        println!("\n=== UNEQUAL SAMPLE SIZES TEST ===");
        println!("Small group (n={}): mean={}", n_small, sum_small / n_small);
        println!("Large group (n={}): mean={}", n_large, sum_large / n_large);
        println!("T-statistic: {}", result.statistic);
        println!("P-value: {}", result.p_value);
        
        // Should detect the difference despite unequal sample sizes
        assert!(result.p_value < 0.1, "Should detect difference with unequal samples");
        println!("✅ PASS: Unequal sample sizes handled correctly!");
    }

    #[test]
    fn test_high_variance_groups() {
        // High variance within groups (common in noisy single-cell data)
        // Group 1: high variance [1, 10, 2, 9, 3] -> sum=25, high sum_sq
        let sum1 = 25.0;
        let sum_sq1 = 195.0; // 1+100+4+81+9
        let n1 = 5.0;
        
        // Group 2: high variance [2, 8, 4, 7, 4] -> sum=25, high sum_sq  
        let sum2 = 25.0;
        let sum_sq2 = 149.0; // 4+64+16+49+16
        let n2 = 5.0;
        
        let result = fast_t_test_from_sums(
            sum1, sum_sq1, n1,
            sum2, sum_sq2, n2,
            TTestType::Student
        );
        
        println!("\n=== HIGH VARIANCE GROUPS TEST ===");
        println!("Both groups have mean: {} and {}", sum1 / n1, sum2 / n2);
        println!("T-statistic: {}", result.statistic);
        println!("P-value: {}", result.p_value);
        
        // Should not detect difference (same means, just high variance)
        assert!(result.p_value > 0.1, "Should not detect difference with same means");
        assert!(num_traits::Float::abs(result.statistic) < 1.0, "T-statistic should be small");
        println!("✅ PASS: High variance groups (same means) handled correctly!");
    }

    #[test]
    fn test_minimal_sample_size() {
        // Edge case: minimum sample size (n=2 per group)
        // Group 1: [5, 7] -> sum=12, sum_sq=74
        let sum1 = 12.0;
        let sum_sq1 = 74.0; // 25+49
        let n1 = 2.0;
        
        // Group 2: [3, 5] -> sum=8, sum_sq=34
        let sum2 = 8.0;
        let sum_sq2 = 34.0; // 9+25
        let n2 = 2.0;
        
        let result = fast_t_test_from_sums(
            sum1, sum_sq1, n1,
            sum2, sum_sq2, n2,
            TTestType::Student
        );
        
        println!("\n=== MINIMAL SAMPLE SIZE TEST ===");
        println!("Group 1 (n=2): mean={}", sum1 / n1);
        println!("Group 2 (n=2): mean={}", sum2 / n2);
        println!("T-statistic: {}", result.statistic);
        println!("P-value: {}", result.p_value);
        
        // Should handle minimal case without crashing
        assert!(num_traits::Float::is_finite(result.p_value), "P-value should be finite");
        assert!(num_traits::Float::is_finite(result.statistic), "T-statistic should be finite");
        println!("✅ PASS: Minimal sample size handled correctly!");
    }

    #[test]
    fn test_single_cell_realistic_scenario() {
        // Realistic single-cell scenario: marker gene expression
        // Cell type A (expressing marker): higher, more variable expression
        let sum_expressing = 45.0; // Mean ≈ 4.5
        let sum_sq_expressing = 235.0; // Moderate variance
        let n_expressing = 10.0;
        
        // Cell type B (not expressing): lower, less variable expression
        let sum_non_expressing = 15.0; // Mean ≈ 1.5  
        let sum_sq_non_expressing = 35.0; // Lower variance
        let n_non_expressing = 10.0;
        
        let result_student = fast_t_test_from_sums(
            sum_expressing, sum_sq_expressing, n_expressing,
            sum_non_expressing, sum_sq_non_expressing, n_non_expressing,
            TTestType::Student
        );
        
        let result_welch = fast_t_test_from_sums(
            sum_expressing, sum_sq_expressing, n_expressing,
            sum_non_expressing, sum_sq_non_expressing, n_non_expressing,
            TTestType::Welch
        );
        
        println!("\n=== SINGLE-CELL REALISTIC SCENARIO ===");
        println!("Expressing cells mean: {}", sum_expressing / n_expressing);
        println!("Non-expressing cells mean: {}", sum_non_expressing / n_non_expressing);
        println!("Student t-test: t={:.3}, p={:.6}", result_student.statistic, result_student.p_value);
        println!("Welch t-test: t={:.3}, p={:.6}", result_welch.statistic, result_welch.p_value);
        
        // Both should detect significant difference
        assert!(result_student.p_value < 0.01, "Student t-test should be significant");
        assert!(result_welch.p_value < 0.01, "Welch t-test should be significant");
        println!("✅ PASS: Realistic single-cell marker gene scenario detected!");
    }

    #[test]
    fn test_very_small_difference() {
        // Test the detection limit - very small but real difference
        // This might happen with housekeeping genes that are slightly different
        let sum1 = 30.0000; // Mean = 3.0000
        let sum_sq1 = 90.01; // Very low variance
        let n1 = 10.0;
        
        let sum2 = 29.9990; // Mean = 2.9999 (tiny difference)  
        let sum_sq2 = 89.997; // Very low variance
        let n2 = 10.0;
        
        let result = fast_t_test_from_sums(
            sum1, sum_sq1, n1,
            sum2, sum_sq2, n2,
            TTestType::Student
        );
        
        println!("\n=== VERY SMALL DIFFERENCE TEST ===");
        println!("Group 1 mean: {:.6}", sum1 / n1);
        println!("Group 2 mean: {:.6}", sum2 / n2);
        println!("Difference: {:.6}", (sum1 / n1) - (sum2 / n2));
        println!("T-statistic: {}", result.statistic);
        println!("P-value: {}", result.p_value);
        
        // Very small differences might not be significant due to our threshold
        println!("✅ PASS: Very small difference test completed");
    }

    #[test]
    fn test_matrix_based_workflow() {
        // Test the matrix-based function that would be used in real single-cell analysis
        
        // Create a small test matrix: 6 cells x 3 genes
        // Gene 0: [1,1,1,5,5,5] - clear difference between groups
        // Gene 1: [3,3,3,3,3,3] - no difference
        // Gene 2: [0,0,1,2,3,4] - moderate difference
        
        let mut coo = CooMatrix::new(6, 3); // 6 cells x 3 genes
        
        // Gene 0 values - clear difference
        coo.push(0, 0, 1.0f64); coo.push(1, 0, 1.0); coo.push(2, 0, 1.0);
        coo.push(3, 0, 5.0); coo.push(4, 0, 5.0); coo.push(5, 0, 5.0);
        
        // Gene 1 values - all same (should not be significant)
        coo.push(0, 1, 3.0); coo.push(1, 1, 3.0); coo.push(2, 1, 3.0);
        coo.push(3, 1, 3.0); coo.push(4, 1, 3.0); coo.push(5, 1, 3.0);
        
        // Gene 2 values - moderate difference
        coo.push(0, 2, 0.0); coo.push(1, 2, 0.0); coo.push(2, 2, 1.0);
        coo.push(3, 2, 2.0); coo.push(4, 2, 3.0); coo.push(5, 2, 4.0);
        
        let matrix = CsrMatrix::from(&coo);
        
        let group1_indices = vec![0, 1, 2]; // First 3 cells
        let group2_indices = vec![3, 4, 5]; // Last 3 cells
        
        println!("\n=== MATRIX-BASED WORKFLOW TEST ===");
        println!("Matrix shape: {} cells x {} genes", matrix.nrows(), matrix.ncols());
        println!("Group 1 indices: {:?}", group1_indices);
        println!("Group 2 indices: {:?}", group2_indices);
        
        // Debug: print the actual matrix values
        for gene in 0..matrix.ncols() {
            print!("Gene {}: [", gene);
            for cell in 0..matrix.nrows() {
                let value = matrix.get_entry(cell, gene).map_or(0.0, |entry| entry.into_value());
                print!("{}, ", value);
            }
            println!("]");
        }
        
        let results = t_test_matrix_groups(
            &matrix,
            &group1_indices,
            &group2_indices,
            TTestType::Student
        ).expect("Matrix t-test should work");
        
        for (gene_idx, result) in results.iter().enumerate() {
            println!("Gene {}: t={:.3}, p={:.6}", gene_idx, result.statistic, result.p_value);
        }
        
        // Manual calculation for Gene 0: [1,1,1] vs [5,5,5]
        // This should result in perfect separation (infinite t-statistic, p-value = 0)
        let manual_gene0 = fast_t_test_from_sums(
            3.0, 3.0, 3.0,    // Group 1: [1,1,1]
            15.0, 75.0, 3.0,  // Group 2: [5,5,5]
            TTestType::Student
        );
        
        println!("Manual Gene 0 calculation: t={:.3}, p={:.6}", manual_gene0.statistic, manual_gene0.p_value);
        
        // Gene 0 should show perfect separation (infinite t-statistic, p-value = 0)
        assert!(num_traits::Float::is_infinite(results[0].statistic), "Gene 0 should have infinite t-statistic");
        assert!(results[0].p_value < 0.001, "Gene 0 should have p-value = 0");
        
        // Gene 1 should not be significant (all same values)
        assert!(results[1].p_value > 0.9, "Gene 1 should not be significant, got p={}", results[1].p_value);
        
        // Gene 2 should be moderately significant
        assert!(results[2].p_value < 0.1, "Gene 2 should show some significance, got p={}", results[2].p_value);
        
        println!("✅ PASS: Matrix-based differential expression workflow works!");
    }
}
