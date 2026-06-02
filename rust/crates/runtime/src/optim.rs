//! Optimizers driving VQE. Two are shipped behind the same `Minimizer`
//! trait so VQE doesn't care which one runs:
//!
//! * [`NelderMead`] — gradient-free downhill simplex. Robust on noisy
//!   expectation values and a good default for low parameter counts.
//! * [`Lbfgs`] — limited-memory BFGS with analytic parameter-shift
//!   gradients. Scales to the larger parameter counts where the simplex
//!   method stalls (the PhysicsVQE / HardwareVQE regime).
//!
//! Both consume the identical `f: params -> energy` closure; L-BFGS gets
//! its gradients from [`parameter_shift_gradient`], which is built on top
//! of that same closure, so swapping optimizers never touches the
//! objective.

use std::collections::VecDeque;
use std::f64::consts::FRAC_PI_2;

pub trait Minimizer {
    /// Minimize `f` starting from `x0`. Returns `(x_min, f_min, n_iters)`.
    fn minimize(&self, x0: Vec<f64>, f: &mut dyn FnMut(&[f64]) -> f64) -> (Vec<f64>, f64, usize);
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
fn l2_norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Exact analytic gradient of an expectation-value objective via the
/// **parameter-shift rule**:
///
/// ```text
/// ∂f/∂θ_i = ½ · [ f(θ + (π/2)·e_i) − f(θ − (π/2)·e_i) ]
/// ```
///
/// This is *exact* — not a finite-difference approximation — for any
/// ansatz whose every parameter feeds a single-qubit Pauli rotation
/// (RX/RY/RZ), since each such generator has eigenvalues ±½. The
/// hardware-efficient ansatz satisfies this. Costs `2N` objective
/// evaluations (each call to `f` here flows through the same per-eval
/// callback VQE wired into the closure, so the gradient probes still
/// stream progress and honour cancellation).
pub fn parameter_shift_gradient(x: &[f64], f: &mut dyn FnMut(&[f64]) -> f64) -> Vec<f64> {
    let mut grad = vec![0.0; x.len()];
    let mut probe = x.to_vec();
    for i in 0..x.len() {
        let xi = probe[i];
        probe[i] = xi + FRAC_PI_2;
        let plus = f(&probe);
        probe[i] = xi - FRAC_PI_2;
        let minus = f(&probe);
        probe[i] = xi;
        grad[i] = 0.5 * (plus - minus);
    }
    grad
}

/// Limited-memory BFGS with a backtracking Armijo line search. The inverse
/// Hessian is applied implicitly by the standard two-loop recursion over
/// the last `history_size` `(s, y)` pairs; the leading scale uses the
/// Barzilai-Borwein-style `γ = ⟨s,y⟩/⟨y,y⟩`. Gradients come from
/// [`parameter_shift_gradient`].
///
/// Termination: `‖∇f‖₂ < gtol`, a failed line search (no Armijo-sufficient
/// decrease — treated as a stationary point), or `max_iters` outer steps.
#[derive(Debug, Clone)]
pub struct Lbfgs {
    pub max_iters: usize,
    /// Stop when the gradient 2-norm drops below this.
    pub gtol: f64,
    /// Number of `(s, y)` curvature pairs retained.
    pub history_size: usize,
    /// Max backtracking halvings per line search.
    pub max_line_search: usize,
}

impl Default for Lbfgs {
    fn default() -> Self {
        Self {
            max_iters: 300,
            gtol: 1e-6,
            history_size: 10,
            max_line_search: 25,
        }
    }
}

impl Minimizer for Lbfgs {
    fn minimize(&self, x0: Vec<f64>, f: &mut dyn FnMut(&[f64]) -> f64) -> (Vec<f64>, f64, usize) {
        let n = x0.len();
        assert!(n >= 1);

        let mut x = x0;
        let mut fx = f(&x);
        let mut g = parameter_shift_gradient(&x, f);

        // Newest at the back; trimmed from the front once full.
        let mut s_hist: VecDeque<Vec<f64>> = VecDeque::with_capacity(self.history_size);
        let mut y_hist: VecDeque<Vec<f64>> = VecDeque::with_capacity(self.history_size);
        let mut rho_hist: VecDeque<f64> = VecDeque::with_capacity(self.history_size);

        let mut iters = 0;
        while iters < self.max_iters {
            if l2_norm(&g) < self.gtol {
                break;
            }

            // Two-loop recursion → d = -H_k ∇f.
            let m = s_hist.len();
            let mut q = g.clone();
            let mut alpha = vec![0.0; m];
            for i in (0..m).rev() {
                let a = rho_hist[i] * dot(&s_hist[i], &q);
                alpha[i] = a;
                for j in 0..n {
                    q[j] -= a * y_hist[i][j];
                }
            }
            let gamma = if m > 0 {
                let yy = dot(&y_hist[m - 1], &y_hist[m - 1]);
                if yy > 0.0 {
                    dot(&s_hist[m - 1], &y_hist[m - 1]) / yy
                } else {
                    1.0
                }
            } else {
                1.0
            };
            for qj in q.iter_mut() {
                *qj *= gamma;
            }
            for i in 0..m {
                let beta = rho_hist[i] * dot(&y_hist[i], &q);
                for j in 0..n {
                    q[j] += (alpha[i] - beta) * s_hist[i][j];
                }
            }
            let mut d: Vec<f64> = q.iter().map(|v| -v).collect();

            // Guard against a non-descent (or NaN) direction — fall back to
            // steepest descent so a degenerate curvature estimate can't stall
            // the whole run. The negation is deliberate: `!(gd < 0.0)` also
            // catches `gd == NaN` (where `gd >= 0.0` would not), so a NaN
            // curvature estimate still triggers the fallback.
            let mut gd = dot(&g, &d);
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            let not_descent = !(gd < 0.0);
            if not_descent {
                d = g.iter().map(|v| -v).collect();
                gd = dot(&g, &d);
            }

            // Backtracking Armijo line search.
            const C1: f64 = 1e-4;
            let mut t = 1.0;
            let mut x_new = x.clone();
            let mut f_new = fx;
            let mut accepted = false;
            for _ in 0..self.max_line_search {
                for j in 0..n {
                    x_new[j] = x[j] + t * d[j];
                }
                f_new = f(&x_new);
                if f_new <= fx + C1 * t * gd {
                    accepted = true;
                    break;
                }
                t *= 0.5;
            }
            if !accepted {
                // No sufficient decrease along a descent direction → treat as
                // a stationary point and stop.
                break;
            }

            let g_new = parameter_shift_gradient(&x_new, f);
            let s: Vec<f64> = (0..n).map(|j| x_new[j] - x[j]).collect();
            let y: Vec<f64> = (0..n).map(|j| g_new[j] - g[j]).collect();
            let sy = dot(&s, &y);
            // Only store curvature pairs that keep the implicit Hessian
            // positive-definite (the standard ⟨s,y⟩ > 0 skip rule).
            if sy > 1e-10 {
                if s_hist.len() == self.history_size {
                    s_hist.pop_front();
                    y_hist.pop_front();
                    rho_hist.pop_front();
                }
                rho_hist.push_back(1.0 / sy);
                s_hist.push_back(s);
                y_hist.push_back(y);
            }

            x = x_new;
            fx = f_new;
            g = g_new;
            iters += 1;
        }

        (x, fx, iters)
    }
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
    fn minimize(&self, x0: Vec<f64>, f: &mut dyn FnMut(&[f64]) -> f64) -> (Vec<f64>, f64, usize) {
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
                let pivot = if use_outside {
                    &reflected
                } else {
                    &simplex[worst]
                };
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
                            simplex[i][j] =
                                best_vertex[j] + sigma * (simplex[i][j] - best_vertex[j]);
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

    #[test]
    fn parameter_shift_matches_analytic_cosine() {
        // Single RY(θ) on |0⟩ measured in Z gives ⟨Z⟩ = cos θ, whose exact
        // derivative is −sin θ. The parameter-shift rule must reproduce it to
        // machine precision (it is exact for Pauli rotations, not an
        // approximation), so we use a tight epsilon at several angles.
        let mut f = |x: &[f64]| x[0].cos();
        for &theta in &[0.0, 0.3, 1.0, 2.5, -1.7] {
            let g = parameter_shift_gradient(&[theta], &mut f);
            assert_abs_diff_eq!(g[0], -theta.sin(), epsilon = 1e-12);
        }
    }

    #[test]
    fn lbfgs_minimizes_quadratic_bowl() {
        // Same bowl as Nelder-Mead; parameter-shift on a smooth quadratic is
        // a central difference at ±π/2, exact for a quadratic, so L-BFGS
        // should land essentially on the analytic minimum (3, -1).
        let opt = Lbfgs::default();
        let mut f = |x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] + 1.0).powi(2);
        let (xs, fmin, iters) = opt.minimize(vec![0.0, 0.0], &mut f);
        assert_abs_diff_eq!(xs[0], 3.0, epsilon = 1e-6);
        assert_abs_diff_eq!(xs[1], -1.0, epsilon = 1e-6);
        assert!(fmin < 1e-10);
        // A 2-D quadratic should converge in a handful of L-BFGS steps.
        assert!(iters <= 25, "took {iters} iters on a quadratic");
    }

    #[test]
    fn lbfgs_minimizes_anisotropic_quadratic_12d() {
        // 12 dims > history_size (10) so the curvature ring buffer trims; an
        // anisotropic well (condition number 100) stresses the implicit
        // Hessian. The ±π/2 parameter-shift stencil is exact-direction on a
        // quadratic, and the minimum coincides with the analytic one, so we
        // can demand a tight result.
        let n = 12;
        let weights: Vec<f64> = (0..n)
            .map(|i| 1.0 + 9.0 * (i as f64 / (n - 1) as f64))
            .collect();
        let center: Vec<f64> = (0..n).map(|i| 0.5 * (i as f64).cos()).collect();
        let w = weights.clone();
        let c = center.clone();
        let mut f = move |x: &[f64]| (0..n).map(|i| w[i] * (x[i] - c[i]).powi(2)).sum::<f64>();
        let opt = Lbfgs {
            max_iters: 500,
            gtol: 1e-10,
            ..Lbfgs::default()
        };
        let (xs, fmin, _) = opt.minimize(vec![0.0; n], &mut f);
        for i in 0..n {
            assert_abs_diff_eq!(xs[i], center[i], epsilon = 1e-5);
        }
        assert!(fmin < 1e-9, "fmin {fmin}");
    }
}
