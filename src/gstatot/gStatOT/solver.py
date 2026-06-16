import jax
import jax.numpy as jnp
from jax import vmap
from jax import jit
from jaxopt import LBFGS
import optax

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
    return exp_gamma.reshape(-1, 1) * (point_kernel_e3 * exp_rho.reshape(1, -1))

gKr_vmap = jax.vmap(gKr, in_axes=(0, 0, None))

# primal transport plan
def pi(F, lam_psi_t, lam_phi_t_1, h, C,  wgamma, epsilon2):

    sol = jnp.exp((-C - (lam_psi_t.reshape(1, -1) + lam_phi_t_1.reshape(1, -1)) 
                   + F - wgamma.reshape(1, -1) + h) / epsilon2)
    return sol 

pi_vmap = vmap(pi, in_axes=(0, 0, 0, 0, None, 0, None))


def direct_sum(a, b):
    return a[:, None] + b

direct_sum_vmap = jax.jit(vmap(direct_sum, in_axes=(0, 0), out_axes=0))

@jit
def point_kernel(C, epsilon2): 
    return jnp.exp(-  C /  epsilon2)

def reg_kernel(psi,phi,C, epsilon1):
    return jnp.exp((psi.reshape(-1, 1) + phi.reshape(1,-1) - C)/epsilon1)

reg_kernel_vmap = jax.vmap(reg_kernel, in_axes=(0,0, None, None))

def fast_objective_and_grad(psi, phi, f, gamma, rho, h, g, col_t, 
                            lam, w, epsilon1, epsilon2, epsilon3, 
                            point_kernel_e1, rC, C, n):

    zeros = jnp.zeros((1, n))
    F = direct_sum_vmap(f, -g * f) 

    wgamma = w * gamma #[:, None, :] * jnp.ones((1, 1, n)) 
    lam_phi_vm = jnp.concatenate([zeros, lam * phi])
    lam_psi_vm = jnp.concatenate([lam * psi, zeros])

    pi_array = pi_vmap(F, lam_psi_vm, lam_phi_vm, h, rC, wgamma, epsilon2)

    psiKphi = reg_kernel_vmap(psi, phi, C, epsilon1) 
    gKr = reg_kernel_vmap(gamma, rho, C, epsilon3) 

    psi_grad = (lam * (pi_array[:-1].sum(axis=1) - psiKphi.sum(axis=2))).ravel() #psi_grad_full(pi_array, psiKphi, lam, n).ravel()
    phi_grad = phi_grad_full(pi_array, psiKphi, lam, n).ravel()
    f_grad = f_grad_vmap(pi_array, g).ravel()
    h_grad = h_grad_vmap(pi_array).ravel()
    gamma_grad = w * (pi_array.sum(axis=1) - gKr.sum(axis=2)).ravel()
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
#    return psi_grad, phi_grad, f_grad, gamma_grad, rho_grad, h_grad, S, grad




def unpack_Y(Y, T=None, n=None):

    psi = Y[:(T-1) * n].reshape((T-1, n))
    phi = Y[(T-1) * n : 2 *(T-1)* n].reshape((T-1, n))
    f = Y[2*(T-1)*n : 2*(T-1)*n + T * n].reshape((T, n))
    gamma= Y[2*(T-1)*n + T * n : 2*(T-1)*n + 2* (T * n)].reshape((T, n))
    rho = Y[2*(T-1)*n + 2*(T * n): 2*(T-1)*n + 3*(T * n)].reshape((T, n))
    h = Y[2*(T-1)*n + 3*(T * n):].reshape((T, ))

    return psi, phi, f, gamma, rho, h


#### BCA solver
from jax.nn import logsumexp


def rho_grad_sol(g2_t, gamma_t, C, epsilon3):

    M = (gamma_t.reshape(-1, 1) - C) / epsilon3
    lse_term = logsumexp(M, axis=0)
    return epsilon3 * jnp.log(g2_t + 1e-15) - epsilon3 * lse_term

########
rho_grad_sol_vmap = jax.vmap(rho_grad_sol, in_axes=(0, 0, None, None))



def h_grad_sol(psi, phi, F,  wgamma, C, epsilon2):
    psi_phi = (psi.reshape(1, -1) + phi.reshape(1, -1))
    wgamma = wgamma.reshape(1, -1)
    B = (-C - psi_phi + F - wgamma)/epsilon2
    return -epsilon2 * logsumexp(B.ravel())

h_grad_sol_vmap = jax.vmap(h_grad_sol, in_axes=(0,0, 0, 0, None, None))


def psi_grad_sol(phi1, phi2, F, wgamma, h2, C, lam, epsilon1, epsilon2, rC):

    B = (-rC - phi1.reshape(1, -1) + F - wgamma.reshape(1, -1) + h2) / epsilon2
    psi2_solution = logsumexp(B, axis=0) - logsumexp((phi2.reshape(1, -1)/(lam + 1e-8) - C) / epsilon1, axis=1)
    psi2_solution = psi2_solution / (1/epsilon1 + lam/epsilon2)

    return psi2_solution 

def phi_grad_sol(phi_t, psi_t, log_pi_tp1, C, lam, epsilon1, epsilon2):
    B = log_pi_tp1 + (lam * (phi_t)) / epsilon2
    v = (psi_t.reshape(1, -1) - C)
    sol = (logsumexp(B, axis=0) - logsumexp(v/epsilon1, axis=1)) / (lam/epsilon2 + 1/epsilon1) # check
    return  sol

phi_grad_sol_vmap = jax.vmap(phi_grad_sol, in_axes=(0, 0, 0, None, 0, None, None))

psi_grad_sol_vmap = jax.vmap(psi_grad_sol, in_axes=(0, 0, 0, 0, 0, None, 0, None, None, None))

def log_pi(F, psi_t, phi_t_1, h, C, wgamma, epsilon2):
    psi_phi = (psi_t.reshape(1, -1) + phi_t_1.reshape(1, -1))
    wgamma = wgamma.reshape(1, -1)
    B = (-C  - psi_phi + F + h - wgamma)/epsilon2
    return B 

log_pi_vmap = jax.vmap(log_pi, in_axes=(0, 0, 0, 0, None, 0, None))

def gamma_grad_sol(rho_t, psi, phi, F, h, w, C, epsilon2, epsilon3, rC):
    B = (-rC - (phi.reshape(1, -1)+ psi.reshape(1,-1)) + F + h) / epsilon2

    return (logsumexp(B,axis=0) - logsumexp((rho_t.reshape(1, -1) - C) / epsilon3, axis=1)) / (w/epsilon2 + 1/epsilon3)

gamma_grad_sol_vmap = jax.vmap(gamma_grad_sol, in_axes=(0, 0, 0, 0, 0, None, None, None, None, None))


@jit
def f_grad_min(f,  g, log_pi_array, epsilon2):
    F = direct_sum_vmap(f, -g * f)
    return logsumexp(log_pi_array + F/epsilon2)

def primal(pi_array, psiKphi, gKr, lam, w, epsilon1, epsilon2, epsilon3, C, rC):
    lam = lam.reshape(-1,1,1)
    reg_sum =  (lam * (C[None,:,:]*psiKphi)).sum() + epsilon1 * (lam * psiKphi * (jnp.log(psiKphi + 1e-15) - 1)).sum()
    fit_sum = w * (C[None,:,:]*gKr).sum() + epsilon3 * w * (gKr * (jnp.log(gKr + 1e-15) - 1)).sum()
    ot_sum = (rC[None,:,:] * pi_array).sum() + epsilon2 * (pi_array * (jnp.log(pi_array + 1e-15) - 1)).sum()

    return reg_sum + fit_sum + ot_sum

def dual(pi_array, psiKphi, gKr, rho, h, col_t, lam, w, epsilon1, epsilon2, epsilon3):

    S = (- epsilon1 * (lam * psiKphi.sum(axis=1)).sum()
         -epsilon2 * pi_array.sum() 
         + (w* rho * col_t).sum() 
         - epsilon3 * w * jnp.sum(gKr) 
         + h.sum()
         )

    return S 


optimizer = optax.lbfgs(learning_rate=1.0)

@jax.jit
def bcd_update_f(f_params, opt_state, g, log_pi_array, epsilon2):
    
    def single_step(carry, _):
        current_f, current_state = carry
        
        def value_fn(f_val):
            return f_grad_min(f_val, g, log_pi_array, epsilon2)
        
        loss, grads = jax.value_and_grad(value_fn)(current_f)
        
        updates, next_state = optimizer.update(grads, current_state, current_f, 
                                               value=loss, grad=grads, value_fn=value_fn)

        next_f = optax.apply_updates(current_f, updates)
        
        return (next_f, next_state), loss

    (new_f_params, new_opt_state), _ = single_step((f_params, opt_state), None)
    
    return new_f_params, new_opt_state

def BCA_step(i, psi, phi,  gamma, rho, h, f,col_t, g, lam, w, epsilon1, epsilon2, epsilon3, C, rC, state):

    zeros = jnp.zeros((1, psi.shape[1]))
    F = direct_sum_vmap(f, -g * f) 
    wgamma = w * gamma
    phi_vm = jnp.concatenate([zeros, lam * phi])

    # update psi
    psi = psi_grad_sol_vmap(phi_vm[:-1], phi_vm[1:], F[:-1], 
                                wgamma[:-1], h[:-1], C, lam, 
                                epsilon1, epsilon2, rC)
    # update phi
    psi_vm = jnp.concatenate([lam * psi, zeros])
    log_pi_array = log_pi_vmap(F, psi_vm, phi_vm, h, rC, wgamma, epsilon2)
    phi = phi_grad_sol_vmap(phi, psi, log_pi_array[1:], C, lam, epsilon1, epsilon2) 
    phi_vm = jnp.concatenate([zeros, lam * phi])

    # update rho
    rho = rho_grad_sol_vmap(col_t, gamma, C, epsilon3) # nonbridge version

    # update gamma
    gamma = gamma_grad_sol_vmap(rho, psi_vm, phi_vm, F, h, w, C, epsilon2, epsilon3, rC)
    wgamma = w * gamma

    # update h
    h = h_grad_sol_vmap(psi_vm, phi_vm, F, wgamma, rC, epsilon2)

    # take a step in f
    log_pi_array = log_pi_vmap(jnp.zeros_like(F), psi_vm, phi_vm, h, rC, wgamma, epsilon2)
    f, state = bcd_update_f(f, state, g, log_pi_array, epsilon2)
    
    return psi, phi, rho, gamma, h, f, state

class jaxSolver:

    def __init__(self, lam, epsilon1, epsilon2, epsilon3, w, r, C, col_t, g, T, N, ages, solver_type='BCA', objective='W2'):

        self.dtype = C.dtype
        self.solver_type = solver_type
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.epsilon3 = epsilon3
        self.C = C

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

        if objective == 'W2':
            self.objective = fast_objective_and_grad
            self.value_and_grad_wrapper = jit(lambda Y: self.vg_wrapper(Y, epsilon2=self.epsilon2))

        self.get_pi_from_Y = jit(lambda Y: self.get_primal(Y, epsilon2=self.epsilon2))

    def vg_wrapper(self, Y, epsilon2=None):

        psi, phi, f, gamma, rho, h = unpack_Y(Y, T=self.T, n=self.n)

        val, grad = self.objective(psi, phi, f, gamma, rho, h,                                                           
                                   self.g, self.col_t, self.lam, self.w, 
                                   self.epsilon1, epsilon2, self.epsilon3, 
                                   0, 
                                   self.r * self.C, 
                                   self.C, self.n)

        return - val, -grad

    def get_primal(self, Y, epsilon2=None):
    
        psi, phi, f, gamma, rho, h = unpack_Y(Y, T=self.T, n=self.n)

        n = self.n
        g = self.g
        w = self.w
        rC = self.r * self.C
        lam = self.lam

        zeros = jnp.zeros((1, n))
        F = direct_sum_vmap(f, -g * f) 
        wgamma = w * gamma

        phi_vm = jnp.concatenate([zeros, lam * phi])
        psi_vm = jnp.concatenate([lam * psi, zeros])
        pi_array = pi_vmap(F, psi_vm, phi_vm, h, rC, wgamma, epsilon2)

        return pi_array

    def solve(self, Y0=None, max_iter=20_000, tol=1e-5, verbose=False, **solver_kwargs):

        if self.solver_type == 'LBFGS':
            return self.LBFGS_solve_with_custom_loop(Y0=Y0, max_iter=max_iter, constraint_tol=tol, verbose=verbose, **solver_kwargs)
        elif self.solver_type == 'BCA':
            return self.BCA_solve(Y0=Y0, max_iter=max_iter, constraint_tol=tol, verbose=verbose, **solver_kwargs)

        else:
            raise ValueError(f"Unsupported solver type: {self.solver_type}. Choose 'LBFGS' or 'BCA'.")
        
    def BCA_solve(self, Y0=None, max_iter=20_000, constraint_tol=1e-4, verbose=False, **solver_kwargs):
        col_t = self.col_t
        g = self.g
        lam = self.lam
        w = self.w
        epsilon1 = self.epsilon1
        epsilon2 = self.epsilon2
        epsilon3 = self.epsilon3
        C = self.C
        r = self.r

        inner_iter = solver_kwargs.get('inner_iter', 100)
        @jit
        def inner_body_fn(i, val):
            psi, phi, gamma, rho, h, f, state = val
            psi_np1, phi_np1, rho_np1, gamma_np1, h_np1, f_np1, state = BCA_step(i, psi, phi, gamma, rho, h, f, col_t, 
                                                                                g, lam , w, 
                                                                                epsilon1, epsilon2, epsilon3, 
                                                                                C, r*C, state)

            return psi_np1, phi_np1, gamma_np1, rho_np1, h_np1, f_np1, state

        @jit
        def run_outer_optimization(init_val, max_iter=100, tol=1e-3, inner_iter=100):

            def cond_fn(loop_state):
                i, gap, *rest = loop_state
                return (i < max_iter) & ((gap > tol) | jnp.isnan(gap)) # gap may be nan if we can't eval the primal/dual objective

            def body_fn_outer(loop_state):
                i, gap, psi, phi, gamma, rho, h, f, state = loop_state

                psi, phi, gamma, rho, h, f, state, = jax.lax.fori_loop(0, inner_iter, inner_body_fn, (psi, phi, gamma, rho, h, f, state))
                rC = r * C
                pi_array = pi_vmap(direct_sum_vmap(f, -g * f), 
                                           jnp.concatenate([lam * psi, jnp.zeros((1, self.n))]), 
                                           jnp.concatenate([jnp.zeros((1, self.n)), lam * phi]), 
                                           h, rC, w * gamma, 
                                           self.epsilon2)

                gKr = reg_kernel_vmap(gamma, rho, C, epsilon3)
                psiKphi = reg_kernel_vmap(psi, phi, C, epsilon1)

                dual_value  = dual(pi_array, psiKphi, gKr, rho, h, col_t, lam, w, epsilon1, epsilon2, epsilon3)
                primal_value = primal(pi_array, psiKphi, gKr, lam, w, epsilon1, epsilon2, epsilon3, C, rC)

                gap = jnp.abs(primal_value - dual_value) / (1 + jnp.abs(primal_value) + jnp.abs(dual_value))

                return (i + 1, gap, psi, phi, gamma, rho, h, f, state)

            psi, phi, gamma, rho, h, f, state = init_val

            init_gap = jnp.inf 
            init_loop_state = (0, init_gap, psi, phi, gamma, rho, h, f, state)

            final_loop_state = jax.lax.while_loop(cond_fn, body_fn_outer, init_loop_state)

            return final_loop_state
        
        if Y0 is None:
            Y0 = 0.01 * jax.random.normal(jax.random.PRNGKey(0), shape=(2*(self.T-1)*self.n + 3*(self.T * self.n) + self.T,), dtype=self.dtype) # random init

        psi, phi, f, gamma, rho, h = unpack_Y(Y0, T=self.T, n=self.n)

        state_init = optimizer.init(f)
        init_val = (psi, phi, gamma, rho, h, f, state_init)

        def run_opt():
            val = run_outer_optimization(init_val, max_iter=max_iter, tol=constraint_tol, inner_iter=inner_iter)
            i, gap, psi, phi, gamma, rho, h, f, state = val
            Y = jnp.concatenate([psi.ravel(), phi.ravel(), gamma.ravel(), rho.ravel(), h.ravel(), f.ravel()])
            pi_array = pi_vmap(direct_sum_vmap(f, -g * f), 
                               jnp.concatenate([lam * psi, jnp.zeros((1, self.n))]), 
                               jnp.concatenate([jnp.zeros((1, self.n)), lam * phi]), 
                               h, r*C, w * gamma, 
                               self.epsilon2)
            max_error = self.test_constraints_vmap(pi_array)

            return Y, pi_array, gap, i, max_error

        run_opt = jax.jit(run_opt)
        return run_opt()
        

    # Deprecated LBFGS
    def LBFGS_solve(self, Y0=None, max_iter=20_000, constraint_tol=1e-5, verbose=False, **solver_kwargs):

        ls = solver_kwargs.get('ls', 'zoom')
        tol = solver_kwargs.get('grad_tol', 1e-10)
        inner_iter = min(solver_kwargs.get('inner_iter', 10_000), max_iter)
        max_linesearch = solver_kwargs.get('max_linesearch', 100)
        min_stepsize = solver_kwargs.get('min_stepsize', 1e-15)
        max_stepsize = solver_kwargs.get('max_stepsize', 4.0)
        verbose_solve = solver_kwargs.get('verbose_solve', False)
        num_restarts = solver_kwargs.get('num_restarts', 5)
        start_w_gd = solver_kwargs.get('start_w_gd', False)

        if Y0 is None:
            Y0 = jnp.zeros(2*(self.T-1)*self.n + 3*(self.T * self.n) + self.T, dtype=self.dtype)
#            0.1 * Y0 =  (jax.random.normal(jax.random.PRNGKey(0), shape=(2*(self.T-1)*self.n + 3*(self.T * self.n) + self.T,)) * 0.01).astype(self.dtype)
            
            if start_w_gd:
                print("Starting with GD warm-up...")
                Y0, _, _, _, _ = self.GD_solve(Y0=Y0, max_iter=inner_iter, constraint_tol=constraint_tol*10, verbose=False, **solver_kwargs)
                print("GD warm-up completed. Starting LBFGS...")


        solver = LBFGS(fun=self.value_and_grad_wrapper, value_and_grad=True, maxiter=inner_iter, verbose=verbose_solve,
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
                Y_opt= (jax.random.normal(jax.random.PRNGKey(n_restart), shape=Y0.shape) * 0.01).astype(self.dtype)
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

        value, grad = self.value_and_grad_wrapper(Y_opt)
        grad_norm = jnp.linalg.norm(grad)

        return Y_opt, value, grad_norm, ran_iter, error
    
    def test_constraints_vmap(self, pi_array):

            g = self.g

            row_sum = jnp.sum(pi_array, axis=2)
            column_sum = jnp.sum(pi_array, axis=1)
            constraint_error = jnp.max(jnp.abs(row_sum - g[:, None] * column_sum))
            pi_sum_error = jnp.max(jnp.abs(jnp.sum(pi_array, axis=(1,2)) - 1))
            return jnp.maximum(constraint_error, pi_sum_error) 

    # attepmt regularized path optimization with LBFGS  
    def LBFGS_solve_with_custom_loop(self, Y0=None, max_iter=20_000, constraint_tol=1e-5, verbose=False, **solver_kwargs):

        E = jax.jit(lambda Y, reg_param: self.test_constraints_vmap(self.get_primal(Y, epsilon2=reg_param)))
#        E = jax.jit(lambda Y, reg_param: self.duality_gap(Y))
        obj = jax.jit(lambda Y, reg_param: self.value_and_grad_wrapper(Y))

        if Y0 is None:
            Y0 = jnp.zeros(2*(self.T-1)*self.n + 3*(self.T * self.n) + self.T, dtype=self.dtype)
#            Y0 = 0.01 * jax.random.normal(jax.random.PRNGKey(0), shape=(2*(self.T-1)*self.n + 3*(self.T * self.n) + self.T,), dtype=self.dtype) # random init

        optimize_fn = build_reg_path_optimizer(obj, E, max_iters=max_iter, e_tol=constraint_tol)

        Y_opt, ran_iter, error = optimize_fn(Y0, jnp.array([self.epsilon2]))  # Assuming a single regularization parameter for now

        value, grad = self.value_and_grad_wrapper(Y_opt[-1])
        dual_value = -value
        pi_array = self.get_pi_from_Y(Y_opt[-1])
        psi, phi, f, gamma, rho, h = unpack_Y(Y_opt[-1], T=self.T, n=self.n)
        gKr = reg_kernel_vmap(gamma, rho, self.C, self.epsilon3)
        psiKphi = reg_kernel_vmap(psi, phi, self.C, self.epsilon1)
        dual_value  = dual(pi_array, psiKphi, gKr, rho, h, self.col_t, self.lam, self.w, self.epsilon1, self.epsilon2, self.epsilon3)
        primal_value = primal(pi_array, psiKphi, gKr, self.lam, self.w, self.epsilon1, self.epsilon2, self.epsilon3, self.C, self.r * self.C)
        gap = jnp.abs(primal_value - dual_value) / (1 + jnp.abs(primal_value) + jnp.abs(dual_value))

        return Y_opt[-1], pi_array, gap, ran_iter[-1], error[-1]



def build_reg_path_optimizer(objective_fn, E, max_iters=1000, e_tol=1e-4):
    lbfgs = LBFGS(fun=objective_fn, value_and_grad=True)

    @jax.jit
    def optimize_path(init_params, reg_params_array):
        
        def path_step(current_carry_params, current_reg_param):
            
            init_opt_state = lbfgs.init_state(current_carry_params, current_reg_param)
            init_error = E(current_carry_params, current_reg_param)
            
            init_loop_state = (current_carry_params, init_opt_state, 0, init_error)

            def cond_fun(state):
                _, _, step, current_error = state
                return (step < max_iters) & ((current_error > e_tol))

            def body_fun(state):
                params, opt_state, step, _ = state
                
                next_params, next_opt_state = lbfgs.update(
                    params, opt_state, current_reg_param
                )
                
                next_error = E(next_params, current_reg_param)
                
                return (next_params, next_opt_state, step + 1, next_error)

            final_state = jax.lax.while_loop(cond_fun, body_fun, init_loop_state)
            final_params, _, final_step, final_error = final_state
            
            return final_params, (final_params, final_step, final_error)


        _, (path_params, path_steps, path_errors) = jax.lax.scan(
            path_step, init_params, reg_params_array
        )
        
        return path_params, path_steps, path_errors

    return optimize_path