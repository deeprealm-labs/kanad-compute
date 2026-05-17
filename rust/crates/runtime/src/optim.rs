//! Lightweight gradient-free optimizers. For VQE bring-up we use the
//! Nelder-Mead downhill-simplex method — robust on noisy quantum
//! expectation values and parameter-free in the sense that only the
//! initial simplex needs to be chosen.
//!
//! When we later wire in `argmin` for L-BFGS / COBYLA / parameter-shift
//! gradients, the `Minimizer` trait stays — VQE doesn't care which
//! optimizer drives it.

pub trait Minimizer {
    /// Minimize `f` starting from `x0`. Returns `(x_min, f_min, n_iters)`.
    fn minimize(
        &self,
        x0: Vec<f64>,
        f: &mut dyn FnMut(&[f64]) -> f64,
    ) -> (Vec<f64>, f64, usize);
}

/// Classical Nelder-Mead with the standard Nash 1990 parameters.
/// Termination: convergence of the simplex (range of vertex f-values
/// below `ftol`) OR `max_iters` reached.
#[derive(Debug, Clone)]
pub struct NelderMead {
    pub max_iters: usize,
    pub ftol: f64,
    pub initial_step: f64,
}

impl Default for NelderMead {
    fn default() -> Self {
        Self {
            max_iters: 2000,
            ftol: 1e-6,
            initial_step: 0.5,
        }
    }
}

impl Minimizer for NelderMead {
    fn minimize(
        &self,
        x0: Vec<f64>,
        f: &mut dyn FnMut(&[f64]) -> f64,
    ) -> (Vec<f64>, f64, usize) {
        let n = x0.len();
        assert!(n >= 1);

        // Initial simplex: x0 plus n shifted copies.
        let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
        simplex.push(x0.clone());
        for i in 0..n {
            let mut v = x0.clone();
            v[i] += self.initial_step;
            simplex.push(v);
        }
        let mut fvals: Vec<f64> = simplex.iter().map(|v| f(v)).collect();
        let mut iters = 0;

        // Standard NM coefficients.
        let alpha = 1.0; // reflection
        let gamma = 2.0; // expansion
        let rho = 0.5; // contraction
        let sigma = 0.5; // shrink

        while iters < self.max_iters {
            // Sort vertices by f-value (best first).
            let mut order: Vec<usize> = (0..=n).collect();
            order.sort_by(|&a, &b| fvals[a].partial_cmp(&fvals[b]).unwrap());
            let best = order[0];
            let worst = order[n];
            let second_worst = order[n - 1];

            // Convergence check.
            let f_range = fvals[worst] - fvals[best];
            if f_range.abs() <= self.ftol {
                break;
            }

            // Centroid of all but the worst.
            let mut centroid = vec![0.0; n];
            for &idx in &order[..n] {
                for j in 0..n {
                    centroid[j] += simplex[idx][j];
                }
            }
            for c in centroid.iter_mut() {
                *c /= n as f64;
            }

            // Reflect.
            let reflected: Vec<f64> = (0..n)
                .map(|j| centroid[j] + alpha * (centroid[j] - simplex[worst][j]))
                .collect();
            let f_reflected = f(&reflected);

            if f_reflected < fvals[order[0]] {
                // Expansion candidate is better than the best so far.
                let expanded: Vec<f64> = (0..n)
                    .map(|j| centroid[j] + gamma * (reflected[j] - centroid[j]))
                    .collect();
                let f_expanded = f(&expanded);
                if f_expanded < f_reflected {
                    simplex[worst] = expanded;
                    fvals[worst] = f_expanded;
                } else {
                    simplex[worst] = reflected;
                    fvals[worst] = f_reflected;
                }
            } else if f_reflected < fvals[second_worst] {
                simplex[worst] = reflected;
                fvals[worst] = f_reflected;
            } else {
                // Contract.
                let use_outside = f_reflected < fvals[worst];
                let pivot = if use_outside { &reflected } else { &simplex[worst] };
                let contracted: Vec<f64> = (0..n)
                    .map(|j| centroid[j] + rho * (pivot[j] - centroid[j]))
                    .collect();
                let f_contracted = f(&contracted);
                let accept = if use_outside {
                    f_contracted <= f_reflected
                } else {
                    f_contracted < fvals[worst]
                };
                if accept {
                    simplex[worst] = contracted;
                    fvals[worst] = f_contracted;
                } else {
                    // Shrink toward best.
                    let best_vertex = simplex[best].clone();
                    for i in 0..=n {
                        if i == best {
                            continue;
                        }
                        for j in 0..n {
                            simplex[i][j] = best_vertex[j] + sigma * (simplex[i][j] - best_vertex[j]);
                        }
                        fvals[i] = f(&simplex[i]);
                    }
                }
            }
            iters += 1;
        }

        // Return best vertex.
        let (best_idx, &best_val) = fvals
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        (simplex[best_idx].clone(), best_val, iters)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn minimizes_quadratic_bowl() {
        // f(x,y) = (x-3)^2 + (y+1)^2; minimum at (3, -1) with f=0.
        let nm = NelderMead::default();
        let mut f = |x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] + 1.0).powi(2);
        let (xs, fmin, _) = nm.minimize(vec![0.0, 0.0], &mut f);
        assert_abs_diff_eq!(xs[0], 3.0, epsilon = 1e-3);
        assert_abs_diff_eq!(xs[1], -1.0, epsilon = 1e-3);
        assert!(fmin < 1e-5);
    }

    #[test]
    fn minimizes_rosenbrock_2d() {
        // Notorious banana — proves NM tracks curved valleys.
        let nm = NelderMead {
            max_iters: 5000,
            ftol: 1e-8,
            initial_step: 0.1,
        };
        let mut f = |x: &[f64]| {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            a * a + 100.0 * b * b
        };
        let (xs, fmin, _) = nm.minimize(vec![-1.2, 1.0], &mut f);
        assert_abs_diff_eq!(xs[0], 1.0, epsilon = 1e-2);
        assert_abs_diff_eq!(xs[1], 1.0, epsilon = 1e-2);
        assert!(fmin < 1e-4);
    }
}
