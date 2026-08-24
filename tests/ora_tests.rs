//! Over-representation analysis.
#![cfg(feature = "enrichment")]

use single_statistics::enrichment::over_representation_analysis;

// ------------------------------------------------------------------------------- ORA

/// Hand-computed: N=100, K=10, n=5, k=5.
/// P(X >= 5) = C(10,5)/C(100,5) = 252 / 75287520 = 3.3472e-6
/// fold enrichment = 5 / (5*10/100) = 10
#[test]
fn ora_matches_hand_calculation() {
    let universe: Vec<usize> = (0..100).collect();
    let pathway: Vec<usize> = (0..10).collect();
    let query: Vec<usize> = (0..5).collect();

    let res = over_representation_analysis(&query, &[pathway], &universe).unwrap();
    assert_eq!(res.len(), 1);
    let r = &res[0];

    assert_eq!(r.overlap, 5);
    assert_eq!(r.pathway_size, 10);
    assert_eq!(r.query_size, 5);
    assert_eq!(r.universe_size, 100);
    assert!((r.fold_enrichment - 10.0).abs() < 1e-9, "fold = {}", r.fold_enrichment);
    assert!(
        (r.p_value - 252.0 / 75_287_520.0).abs() < 1e-12,
        "p = {} (expected {})",
        r.p_value,
        252.0 / 75_287_520.0
    );
}

#[test]
fn ora_reports_no_enrichment_for_a_disjoint_pathway() {
    let universe: Vec<usize> = (0..100).collect();
    let pathway: Vec<usize> = (50..60).collect();
    let query: Vec<usize> = (0..5).collect();

    let res = over_representation_analysis(&query, &[pathway], &universe).unwrap();
    assert_eq!(res[0].overlap, 0);
    assert_eq!(res[0].p_value, 1.0);
    assert_eq!(res[0].fold_enrichment, 0.0);
}

#[test]
fn ora_ignores_genes_outside_the_universe() {
    let universe: Vec<usize> = (0..100).collect();
    // Half the pathway and part of the query sit outside the universe.
    let pathway: Vec<usize> = (0..10).chain(500..510).collect();
    let query: Vec<usize> = (0..5).chain(900..905).collect();

    let res = over_representation_analysis(&query, &[pathway], &universe).unwrap();
    assert_eq!(res[0].pathway_size, 10, "out-of-universe pathway genes must be dropped");
    assert_eq!(res[0].query_size, 5, "out-of-universe query genes must be dropped");
}

#[test]
fn ora_adjusts_across_pathways() {
    let universe: Vec<usize> = (0..200).collect();
    let pathways: Vec<Vec<usize>> = (0..20).map(|i| (i * 10..i * 10 + 10).collect()).collect();
    let query: Vec<usize> = (0..10).collect();

    let res = over_representation_analysis(&query, &pathways, &universe).unwrap();
    assert_eq!(res.len(), 20);
    for r in &res {
        assert!(r.adjusted_p_value >= r.p_value - 1e-12, "BH must not shrink p-values");
        assert!((0.0..=1.0).contains(&r.adjusted_p_value));
    }
    // Pathway 0 is exactly the query set and should be the strongest hit.
    assert_eq!(res[0].overlap, 10);
    assert!(res[0].adjusted_p_value < 0.05);
}

#[test]
fn ora_rejects_empty_inputs() {
    assert!(over_representation_analysis(&[1], &[vec![1]], &[]).is_err());
    assert!(over_representation_analysis(&[1], &[], &[1, 2, 3]).is_err());
}

