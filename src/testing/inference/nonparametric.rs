use crate::testing::{Alternative, TestResult};
use nalgebra_sparse::CsrMatrix;
use num_traits::{Float, NumCast};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use single_utilities::traits::{FloatOps, FloatOpsTS};
use statrs::distribution::{ContinuousCDF, Normal};
use std::cmp::Ordering;

pub fn mann_whitney_matrix_groups<T>(
    matrix: &CsrMatrix<T>,
    group1_indices: &[usize],
    group2_indices: &[usize],
    alternative: Alternative,
) -> anyhow::Result<Vec<TestResult<T>>>
where
    T: FloatOpsTS,
{
    if group1_indices.is_empty() || group2_indices.is_empty() {
        return Err(anyhow::anyhow!("Group indices cannot be empty"));
    }

    let nrows = matrix.nrows();

    let results: Vec<_> = (0..nrows)
        .into_par_iter()
        .map(|row| {
            let mut group1_values: Vec<T> = Vec::with_capacity(group1_indices.len());
            let mut group2_values: Vec<T> = Vec::with_capacity(group2_indices.len());

            for &col in group1_indices {
                if let Some(entry) = matrix.get_entry(row, col) {
                    let value = entry.into_value();
                    group1_values.push(value);
                }
            }

            for &col in group2_indices {
                if let Some(entry) = matrix.get_entry(row, col) {
                    let value = entry.into_value();
                    group2_values.push(value);
                }
            }
            mann_whitney(&group1_values, &group2_values, alternative)
        })
        .collect();

    Ok(results)
}

pub fn mann_whitney<T>(x: &[T], y: &[T], alternative: Alternative) -> TestResult<T>
where
    T: FloatOps,
{
    let nx = x.len();
    let ny = y.len();

    if nx == 0 || ny == 0 {
        return TestResult::new(<T as num_traits::Float>::nan(), T::one()); // Insufficient data
    }

    // Combine samples and assign group labels (0 for x, 1 for y)
    let mut combined: Vec<(T, usize)> = Vec::with_capacity(nx + ny);
    combined.extend(x.iter().map(|&v| (v, 0)));
    combined.extend(y.iter().map(|&v| (v, 1)));

    // Sort by value
    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    // Assign ranks (with ties averaged)
    let mut ranks = vec![T::zero(); nx + ny];
    let mut i = 0;
    while i < combined.len() {
        let val = combined[i].0;
        let mut j = i + 1;

        // Find tied values
        while j < combined.len() && combined[j].0 == val {
            j += 1;
        }

        // Assign average rank to ties
        let rank = T::from(i + j - 1).unwrap() / T::from(2.0).unwrap() + T::one();
        for k in i..j {
            ranks[k] = rank;
        }

        i = j;
    }

    // Calculate rank sum for group X
    let mut rank_sum_x = T::zero();
    for i in 0..combined.len() {
        if combined[i].1 == 0 {
            rank_sum_x += ranks[i];
        }
    }

    let u_x = rank_sum_x - T::from(nx * (nx + 1)).unwrap() / T::from(2.0).unwrap();
    let u_y = T::from(nx * ny).unwrap() - u_x;

    let u = match alternative {
        Alternative::TwoSided => Float::min(u_x, u_y),
        Alternative::Less => u_x,
        Alternative::Greater => u_y,
    };

    let mean_u = T::from(nx * ny).unwrap() / T::from(2.0).unwrap();
    let var_u = T::from(nx * ny * (ny + nx + 1)).unwrap() / T::from(12.0).unwrap();

    let correction = T::from(0.5).unwrap();

    let z = match alternative {
        Alternative::TwoSided => {
            let z_score = (Float::max(u, mean_u) - mean_u - correction) / var_u.sqrt();
            Float::abs(z_score)
        }
        Alternative::Less => (u_x - mean_u + correction) / var_u.sqrt(),
        Alternative::Greater => (u_y - mean_u + correction) / var_u.sqrt(),
    };

    let normal = Normal::new(0.0, 1.0).unwrap();
    let z_f64 = z.to_f64().unwrap();

    let p_value = match alternative {
        Alternative::TwoSided => 2.0 * (1.0 - normal.cdf(z_f64)),
        _ => 1.0 - normal.cdf(z_f64),
    };
    let p_value = T::from(p_value).unwrap();

    let effect_size = z / T::from(nx + ny).unwrap().sqrt();

    // Standard error of U
    let standard_error = var_u.sqrt();

    TestResult::with_effect_size(u, p_value, effect_size)
        .with_standard_error(standard_error)
        .with_metadata("z_score", z)
        .with_metadata("mean_u", mean_u)
        .with_metadata("var_u", var_u)
        .with_metadata("nx", T::from(nx).unwrap())
        .with_metadata("ny", T::from(ny).unwrap())
}
