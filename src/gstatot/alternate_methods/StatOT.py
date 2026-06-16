import os
import jax
from jax.random import key
from matplotlib.pyplot import plot
import jax.numpy as jnp
import numpy as np
import gstatot.utilities.utils as utils 
import pandas as pd
from ott.geometry import geometry
from ott.solvers import linear

class StatOT:
    
    def __init__(self, adata, adata_keys, dt=0.25, 
                 growth_rate_func=None, dtype=jnp.float32, save_dir=None) -> None:

        self.dtype = dtype
        if dtype is jnp.float64:
            jax.config.update("jax_enable_x64", True)

        else:
            jax.config.update("jax_enable_x64", False)

        self.adata = adata
        self.dt = self.dtype(dt) 
        adata_keys_needed = ['time_key', 'embed_key', 'growth_rate_key']

        if not all([key in adata_keys.keys() for key in adata_keys_needed]):
            raise ValueError(f"adata_keys must contain the following keys: {adata_keys_needed}")

        for key in adata_keys.keys():
            setattr(self, key, adata_keys[key])

        self.times = jnp.unique(jnp.asarray(self.adata.obs[self.time_key], dtype=self.dtype))

        self.N = self.adata.shape[0]
        self.T = len(self.times)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True) if save_dir is not None else None


        # growth rates
        if growth_rate_func is not None:
            self.all_growth_rates = [growth_rate_func(self.adata[self.adata.obs[self.time_key] == time].obsm[self.embed_key], time) 
                                               for time in self.times]
        else:
            self.all_growth_rates = [self.adata[self.adata.obs[self.time_key] == time].obs[self.growth_rate_key].to_numpy() for time in self.times]

        self.all_growth = [jnp.exp(growth * self.dt).astype(self.dtype) for growth in self.all_growth_rates]

    def set_save_dir(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True) if save_dir is not None else None

    def fit(self, model_params={}, max_iter=10_000, inner_iter=5_000, verbose=False, constraint_tol=1e-5):

        if 'epsilon' in model_params.keys():
            epsilon = model_params['epsilon']
        else:
            print("Using default epsilon=0.025")
            epsilon = 0.025

        if 'lse' in model_params.keys():
            lse = model_params['lse']
        else:
            print("Using default lse=False")
            lse = False

        if 'cost_scaling' in model_params.keys():
            cost_scaling = model_params['cost_scaling']
            if verbose:
                print('Cost scaled by ', cost_scaling)
        else:
            cost_scaling = 'Mean'
            print("Using default cost_scaling='Mean'")


        solver_res = {'time': [], 'inf_norm_error': []} 
        for i, time in enumerate(self.times):
            X = jnp.array(self.adata[self.adata.obs[self.time_key] == time].obsm[self.embed_key], dtype=self.dtype)
            g = self.all_growth[i]

            a = g/g.sum()
            b = jnp.ones(X.shape[0], dtype=self.dtype) * (1.0 / X.shape[0])

            C = scale_cost(X, scale_type=cost_scaling, verbose=verbose)
            inf_norm_error = jnp.inf
            ran_iter = 0
            while inf_norm_error > constraint_tol and ran_iter < max_iter:
                if lse:
                    pi = sinkhorn_lse(a, b, C, epsilon, inner_iter)
                else:
                    pi = sinkhorn(a, b, C, epsilon, inner_iter)
                inf_norm_error = utils.check_inf_norm_agreement(pi, a, b)
                ran_iter += inner_iter
                if verbose:
                    print(f"Time {time}, ran {ran_iter} iterations, inf norm error: {inf_norm_error:.4e}")

            solver_res['time'].append(time)
            solver_res['inf_norm_error'].append(float(inf_norm_error))
            solver_res['n_iter'] = ran_iter

            self.adata.uns[f'pi_{time}'] = np.array(pi/pi.sum(), dtype=np.float32)
        self.solver_df = pd.DataFrame(solver_res)

        if self.save_dir is not None:
            self.solver_df.to_csv(os.path.join(self.save_dir, 'solver_results.csv'), index=False)


    def get_lin_fate_probs(self, label_key, all_labels=None, lin_fp_error_tol=1e-2):

        utils.get_lin_fate_probs(self, label_key=label_key, all_labels=all_labels, lin_fp_error_tol=lin_fp_error_tol, full_supp=False)

    def get_trajectories(self, num_step, num_traj, key='StatOT_traj_data', 
                         plot_hitting_time=False, plot_traj=False, return_indices=False, plotting_embed='X_pca', 
                         return_traj_len=False):

        if type(num_step) is int:
            num_step_list = [num_step] * self.T
        else:
            num_step_list = num_step
        if type(num_traj) is int:
            num_traj_list = [num_traj] * self.T
        else:
            num_traj_list = num_traj

        transition_matrices = [utils.row_normalize(self.adata.uns[f'pi_{self.times[i]}']) for i in range(self.T)]
        init_dists = [jnp.clip(self.adata[self.adata.obs[self.time_key] == self.times[i]].obs[self.growth_rate_key].to_numpy(), 0, None) for i in range(self.T)]

        traj_data_ind = utils.get_traj_distributions(transition_matrices=transition_matrices,
                                                  init_dists=init_dists,
                                                  num_traj_list=num_traj_list,
                                                  num_step_list=num_step_list)
        test_traj_data = [self.adata[self.adata.obs[self.time_key] == self.times[i]].obsm[self.embed_key][traj_data_ind[i]] for i in range(len(traj_data_ind))]

        self.adata.uns[key] = {}
        for i, age in enumerate(self.times):
            self.adata.uns[key][str(age)] = test_traj_data[i]

        sink_idx_list = [np.where(self.all_growth_rates[i] < 0)[0] for i in range(self.T)]

        if plot_hitting_time:
            traj_len = utils.plot_time_to_sink(self.times,
                                    n_points_list=[len(self.all_growth_rates[i]) for i in range(self.T)],
                                    traj_data_ind=traj_data_ind,
                                    sink_idx_list=sink_idx_list,
                                    return_traj_len=True)

        if plot_traj:
            if self.save_dir is not None:
                utils.plot_trajectories('StatOT', self.adata, N=50, ncols=5, imsize=5,
                                        sup_title=f"StatOT Trajectories", traj_data_ind=traj_data_ind, embed_key=plotting_embed,
                                        save_path=os.path.join(self.save_dir, f'statOT_trajectories.png'))
            else:
                utils.plot_trajectories('StatOT', self.adata, N=50, ncols=5, imsize=5, 
                                        sup_title=f"StatOT Trajectories", traj_data_ind=traj_data_ind, embed_key=plotting_embed,
                                        save_path=None)

        if return_indices:
            return traj_data_ind

        elif return_traj_len:
            return traj_len
    
    def get_mfpt(self):
        utils.get_mfpt(self, full_supp=False, dt=self.dt)
                



def sinkhorn(a, b, cost, epsilon, iterations):

    return linear.solve(
        geometry.Geometry(cost_matrix=cost, epsilon=epsilon),
        a=a,
        b=b,
        lse_mode=False,
        min_iterations=iterations,
        max_iterations=iterations,
    ).matrix

def sinkhorn_lse(a, b, cost, epsilon, iterations, return_duals=False):
    result = linear.solve(
        geometry.Geometry(cost_matrix=cost, epsilon=epsilon),
        a=a,
        b=b,
        lse_mode=True,
        min_iterations=iterations,
        max_iterations=iterations,
    )
    if return_duals:
        return result.matrix, result.dual_a, result.dual_b
    else:
        return result.matrix

sinkhorn = jax.jit(sinkhorn, static_argnums=[3, 4])
sinkhorn_lse = jax.jit(sinkhorn_lse, static_argnums=[3, 4])


def scale_cost(X, scale_type='mean', verbose=False):
    scale_type = scale_type.capitalize() if scale_type is not None else None
    C = utils.vmap_sq_dist(X, X)

    if scale_type == 'Max':
        C = C / jnp.max(C)

    elif scale_type == 'Mean':
        C = C / jnp.mean(C)

    elif scale_type == 'Median':
        C = C / jnp.median(C)

    elif (scale_type != None):
        raise ValueError(f"cost_scaling must be None, 'Max', 'Mean' or 'Median' not {scale_type}")

    return C