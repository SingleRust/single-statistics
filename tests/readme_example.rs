//! Compiles and runs the README's examples. If this breaks, the README is stale.

use single_statistics::testing::markers::{filter_marker_candidates, marker_statistics};
use single_statistics::testing::utils::SprsView;
use single_statistics::testing::{MatrixStatTests, TestMethod, TTestType};
use sprs::{CsMat, TriMat};

/// Gene 0 separates the groups with real within-group spread; the rest are flat.
fn cell_value(gene: usize, cell: usize) -> f64 {
    if gene == 0 {
        if cell < 3 { 1.0 + cell as f64 } else { 8.0 + (cell - 3) as f64 }
    } else {
        3.0
    }
}

fn example_matrix() -> CsMat<f64> {
    // 4 genes x 6 cells
    let mut tri = TriMat::new((4, 6));
    for g in 0..4 {
        for c in 0..6 {
            tri.add_triplet(g, c, cell_value(g, c));
        }
    }
    tri.to_csr()
}

#[test]
fn readme_differential_expression() -> anyhow::Result<()> {
    let expression_matrix: CsMat<f64> = example_matrix();
    let group_ids = vec![0, 0, 0, 1, 1, 1];

    let results = expression_matrix
        .differential_expression(&group_ids, TestMethod::TTest(TTestType::Welch))?;

    let significant_genes = results.significant_indices(0.05);
    println!("Found {} significant genes", significant_genes.len());

    if let Some(effect_sizes) = &results.effect_sizes {
        for (i, &gene_idx) in significant_genes.iter().enumerate() {
            println!(
                "Gene {}: statistic = {}, p-value = {}, adjusted p-value = {}, effect size = {}",
                gene_idx,
                results.statistics[gene_idx],
                results.p_values[gene_idx],
                results.adjusted_p_values.as_ref().unwrap()[gene_idx],
                effect_sizes[i]
            );
        }
    }

    assert!(significant_genes.contains(&0), "gene 0 is a clear marker");
    Ok(())
}

#[test]
fn readme_marker_genes() -> anyhow::Result<()> {
    let expression_matrix = example_matrix();
    let group1 = vec![0, 1, 2];
    let group2 = vec![3, 4, 5];

    let view = SprsView::new(&expression_matrix);
    let stats = marker_statistics(view.as_matrix_ref(), &group1, &group2, 1.0)?;

    let candidates = filter_marker_candidates(&stats, 0.10, 0.25);

    assert_eq!(stats.len(), 4);
    assert!(candidates.contains(&0), "gene 0 should survive filtering");
    assert!(!candidates.contains(&1), "a flat gene should not");
    Ok(())
}

/// The README claims CSR genes x cells and CSC cells x genes are interchangeable.
#[test]
fn readme_orientation_claim() -> anyhow::Result<()> {
    let csr = example_matrix();

    let mut tri = TriMat::new((6, 4));
    for g in 0..4 {
        for c in 0..6 {
            tri.add_triplet(c, g, cell_value(g, c));
        }
    }
    let csc: CsMat<f64> = tri.to_csc();

    let a = csr.t_test(&[0, 1, 2], &[3, 4, 5], TTestType::Welch)?;
    let b = csc.t_test(&[0, 1, 2], &[3, 4, 5], TTestType::Welch)?;

    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        // Flat genes give NaN and perfectly separated ones can give +/-inf, so
        // compare exactly rather than by difference.
        assert_eq!(
            x.statistic.to_bits(), y.statistic.to_bits(),
            "csr t={} vs csc t={}", x.statistic, y.statistic
        );
    }
    assert!(a[0].statistic.is_finite(), "gene 0 should be a finite comparison");
    Ok(())
}
