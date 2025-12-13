import jax
import jax.numpy as jnp
from jax import vmap
from jax import jit
from jaxopt import LBFGS as jaxLBFGS

# dual gradients
def rho_grad(row_t, rhoKgamma, w):
    return w * (row_t - rhoKgamma.sum(axis=0))

rho_grad_vmap = jax.vmap(rho_grad, in_axes=(0, 0, None))

def gamma_grad(pi, rhoKgamma, w):
    return w * (pi.sum(axis=1) - rhoKgamma.sum(axis=1))

gamma_grad_vmap = jax.vmap(gamma_grad, in_axes=(0, 0, None))

def h_grad(pi):
    return 1 - pi.sum()

h_grad_vmap = jax.vmap(h_grad, in_axes=(0,))

def psi_grad_full(pi, psiKphi_array, lam, n):
    return lam * (pi[:-1].sum(axis=1) - psiKphi_array.sum(axis=2))


def phi_grad_full(pi_array, psiKphi_array, lam, n):
    return lam * (pi_array[1:].sum(axis=1) - psiKphi_array.sum(axis=1))

def gamma_grad_col_fit(pi, rhoKgamma, w):
    return w * (pi.sum(axis=0) - rhoKgamma.sum(axis=1))

gamma_grad_col_fit_vmap = jax.vmap(gamma_grad_col_fit, in_axes=(0, 0, None))

def f_grad(pi, g):
    return -pi.sum(axis=1) + g * pi.sum(axis=0)

f_grad_vmap = jax.vmap(f_grad, in_axes=(0, 0))


# dual kernel computations

def psiKphi(exp_psi, exp_phi, point_kernel_eps1, n):
    uKv = exp_psi[:, None] * (point_kernel_eps1 * exp_phi)
    return uKv

psiKphi_vmap = jax.vmap(psiKphi, in_axes=(0, 0, None, None))

def gKr(exp_gamma, exp_rho, point_kernel_e3):
    return exp_gamma[:, None] * (point_kernel_e3 * exp_rho)

gKr_vmap = jax.vmap(gKr, in_axes=(0, 0, None))

# primal transport plan
def pi(F, lam_psi_t, lam_phi_t_1, h, C,  wgamma, epsilon2):

    psi_phi = lam_psi_t + lam_phi_t_1 + jnp.zeros_like(F)
    B = -C  - psi_phi + F - wgamma + h

    return jnp.exp(B / epsilon2) 

pi_vmap = vmap(pi, in_axes=(0, 0, 0, 0, None, 0, None))


def direct_sum(a, b):
    return a[:, None] + b

direct_sum_vmap = jax.jit(vmap(direct_sum, in_axes=(0, 0), out_axes=0))

@jit
def point_kernel(C, epsilon2): 
    return jnp.exp(-  C /  epsilon2)


def fast_objective_and_grad(psi, phi, f, gamma, rho, h, g, col_t, 
                            lam, w, epsilon1, epsilon2, epsilon3, 
                            point_kernel_e1, point_kernel_e3, C, n):

    zeros = jnp.zeros((1, n))
    F = direct_sum_vmap(f, -g * f) 

    wgamma = w * gamma[:, None, :] * jnp.ones((1, 1, n)) 

    lam_phi_vm = jnp.concatenate([zeros, lam * phi])
    lam_psi_vm = jnp.concatenate([lam * psi, zeros])

    pi_array = pi_vmap(F, lam_psi_vm, lam_phi_vm, h, C, wgamma, epsilon2)


    exp_psi =  jnp.exp(psi / epsilon1) 
    exp_phi =  jnp.exp(phi / epsilon1)
    exp_gamma =jnp.exp(gamma / epsilon3)
    exp_rho =  jnp.exp(rho / epsilon3)

    gKr = gKr_vmap(exp_gamma, exp_rho, point_kernel_e3)
    psiKphi = psiKphi_vmap(exp_psi, exp_phi, point_kernel_e1, n)

    # compute grad
    psi_grad = psi_grad_full(pi_array, psiKphi, lam, n).ravel()
    phi_grad = phi_grad_full(pi_array, psiKphi, lam, n).ravel()
    f_grad = f_grad_vmap(pi_array, g).ravel()
    h_grad = h_grad_vmap(pi_array).ravel()
    gamma_grad = gamma_grad_col_fit_vmap(pi_array, gKr, w).ravel()
    rho_grad = rho_grad_vmap(col_t, gKr, w).ravel()

    grad = jnp.concatenate((psi_grad, phi_grad, f_grad, gamma_grad, rho_grad, h_grad))

    # compute objective
    S = (- epsilon1 * (lam * psiKphi.sum(axis=1)).sum()
         -epsilon2 * pi_array.sum() 
         + (w* rho * col_t).sum() 
         - epsilon3 * w * jnp.sum(gKr) 
         + h.sum()
         )

    return S, grad

def unpack_Y(Y, T=None, n=None):

    psi = Y[:(T-1) * n].reshape((T-1, n))
    phi = Y[(T-1) * n : 2 *(T-1)* n].reshape((T-1, n))
    f = Y[2*(T-1)*n : 2*(T-1)*n + T * n].reshape((T, n))
    gamma= Y[2*(T-1)*n + T * n : 2*(T-1)*n + 2* (T * n)].reshape((T, n))
    rho = Y[2*(T-1)*n + 2*(T * n): 2*(T-1)*n + 3*(T * n)].reshape((T, n))
    h = Y[2*(T-1)*n + 3*(T * n):].reshape((T, ))

    return psi, phi, f, gamma, rho, h



class jaxSolver:

    def __init__(self, lam, epsilon1, epsilon2, epsilon3, w, r, C, col_t, g, T, N, ages):

        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.epsilon3 = epsilon3
        self.C = C
        self.rC = r * C

        self.point_kernel_e1 = point_kernel(self.C, epsilon1)
        self.point_kernel_e3 = point_kernel(self.C, epsilon3)

        self.col_t = col_t
        self.g = g
        self.T = T
        self.n = N
        self.w = w
        self.r = r
        self.ages = ages
        self.da = ages[1:] - ages[:-1]
        lam_a = (lam / self.da)[:, None]
        self.lam = lam_a

        self.objective = fast_objective_and_grad

        @jit
        def value_and_grad_wrapper(Y):

            psi, phi, f, gamma, rho, h = unpack_Y(Y, T=T, n=N)

            val, grad = self.objective(psi, phi, f, gamma, rho, h,                                                           
                                       g, col_t, lam_a, w, 
                                       epsilon1, epsilon2, epsilon3, 
                                       self.point_kernel_e1, 
                                       self.point_kernel_e3, 
                                       self.rC, self.n)

            return - val, -grad
        
        self.value_and_grad_wrapper = value_and_grad_wrapper

        @jit
        def get_pi_from_Y(Y):
           
            psi, phi, f, gamma, rho, h = unpack_Y(Y, T=self.T, n=self.n)
    
            n = self.n
            g = self.g
            w = self.w
            C = self.C
            lam = self.lam
            epsilon2 = self.epsilon2
            r = self.r
    
            zeros = jnp.zeros((1, n))
            F = direct_sum_vmap(f, -g * f) 
            wgamma = jnp.broadcast_to(w * gamma[:, None, :], (gamma.shape[0], gamma.shape[1], gamma.shape[1]))
    
            phi_vm = jnp.concatenate([zeros, lam * phi])
            psi_vm = jnp.concatenate([lam * psi, zeros])
            pi_array = pi_vmap(F, psi_vm, phi_vm, h, r*C, wgamma, epsilon2)
    
            return pi_array
        
        self.get_pi_from_Y = get_pi_from_Y


    def solve(self, Y0=None, max_iter=20_000, constraint_tol=1e-5, verbose=False, **solver_kwargs):

        ls = solver_kwargs.get('ls', 'zoom')
        tol = solver_kwargs.get('grad_tol', 1e-10)
        inner_iter = min(solver_kwargs.get('inner_iter', 10_000), max_iter)
        max_linesearch = solver_kwargs.get('max_linesearch', 100)
        min_stepsize = solver_kwargs.get('min_stepsize', 1e-15)
        max_stepsize = solver_kwargs.get('max_stepsize', 4.0)
        verbose_solve = solver_kwargs.get('verbose_solve', False)
        num_restarts = solver_kwargs.get('num_restarts', 5)

        if Y0 is None:
            Y0 = jnp.zeros(2*(self.T-1)*self.n + 3*(self.T * self.n) + self.T)

        solver = jaxLBFGS(fun=self.value_and_grad_wrapper, value_and_grad=True, maxiter=inner_iter, verbose=verbose_solve,
                       max_stepsize=max_stepsize, min_stepsize=min_stepsize, maxls=max_linesearch, stop_if_linesearch_fails=True,
                       linesearch=ls, tol=tol)

        error = jnp.inf
        Y_opt = Y0
        ran_iter = 0
        n_restart = 0
        jit_pi_from_Y = jax.jit(self.get_pi_from_Y)

        while error > constraint_tol:
            sol = solver.run(Y_opt)

            if (ran_iter == 0) & (sol.state.failed_linesearch) & (n_restart < num_restarts):
                print(f"Warning: Line search failed during first {sol.state.iter_num} iterations: Attempting random restart {n_restart+1}/{num_restarts}")
                n_restart += 1
                Y_opt= jax.random.normal(jax.random.PRNGKey(n_restart), shape=Y0.shape) * 0.01
                continue

            Y_opt = sol.params
            ran_iter += sol.state.iter_num
            pi_array = jit_pi_from_Y(Y_opt)
            error, pi_sum_error = self.test_constraints(pi_array)
            error = jnp.maximum(error, pi_sum_error)

            if verbose:
                print(f"Completed {ran_iter} total iterations, current error: {error:.4e}")

            if (ran_iter >= max_iter) or sol.state.failed_linesearch:
                break

        return sol, ran_iter, error



    def test_constraints(self, pi_array):

            g = self.g
            constraint_error = []
            pi_sum_error = []
            for i, pi_t in enumerate(pi_array):
                row_sum = jnp.sum(pi_t, axis=1)
                column_sum = jnp.sum(pi_t, axis=0)

                constraint_error_i = jnp.max(jnp.abs(row_sum - g[i] * column_sum))
                pi_sum_error_i = jnp.abs(jnp.sum(pi_t) - 1)
                constraint_error.append(constraint_error_i)
                pi_sum_error.append(pi_sum_error_i)

            return jnp.max(jnp.asarray(constraint_error)), jnp.max(jnp.asarray(pi_sum_error))
