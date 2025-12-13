import time
from gstatot import solver
import jax
import jax.numpy as jnp
import numpy as np
from gstatot import utils
import pandas as pd
import os


class gStatOT:
    
    def __init__(self, adata, adata_keys, dt=0.25, cost_scaling='mean', 
                 growth_rate_func=None, dtype=jnp.float64, save_dir=None) -> None:

        if dtype == jnp.float64:
            jax.config.update("jax_enable_x64", True)
        self.dtype = dtype

        self.adata = adata
        self.dt = self.dtype(dt) 
        self.cost_scaling = cost_scaling.capitalize()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True) if save_dir is not None else None

        adata_keys_needed = ['time_key', 'embed_key', 'growth_rate_key']

        if not all([key in adata_keys.keys() for key in adata_keys_needed]):
            raise ValueError(f"adata_keys must contain the following keys: {adata_keys_needed}")

        for key in adata_keys.keys():
            setattr(self, key, adata_keys[key])

        self.times = jnp.array(sorted(self.adata.obs[self.time_key].unique()), dtype=self.dtype)
        X = jnp.asarray(self.adata.obsm[self.embed_key], dtype=self.dtype)

        self.N = X.shape[0]
        self.T = len(self.times)
        self.C = utils.vmap_sq_dist(X, X)

        if self.cost_scaling == 'Max':
            self.C = self.C / jnp.max(self.C)
            print("Cost scaled by max")

        elif self.cost_scaling == 'Mean':
            self.C = self.C / jnp.mean(self.C)
            print("Cost scaled by mean")

        elif self.cost_scaling == 'Median':
            self.C = self.C / jnp.median(self.C)
            print("Cost scaled by median")

        elif (self.cost_scaling != None):
            raise ValueError("cost_scaling must be None, 'Max', 'Mean' or 'Median'")

        # growth rates
        if growth_rate_func is not None:
            self.all_growth_rates = jnp.array([growth_rate_func(X, time) for time in self.times], dtype=self.dtype)
        else:
            self.growth_rates = jnp.array(self.adata.obs[self.growth_rate_key], dtype=self.dtype)
            self.all_growth_rates = jnp.array([self.growth_rates for _ in range(self.T)], dtype=self.dtype)

        self.all_growth = jnp.exp(self.all_growth_rates * self.dt)

        data_dists = []
        for t in self.times:
            col_cur = np.zeros(self.N)
            col_cur[self.adata.obs[self.time_key] == t] = 1.0
            col_cur = col_cur / np.sum(col_cur)
            data_dists.append(col_cur)

        self.data_dists = jnp.array(data_dists, dtype=self.dtype)

    def set_save_dir(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True) if save_dir is not None else None

    def fit(self, model_params={}, max_iter=1000, Y0=None, verbose=False, constraint_tol=1e-5, solver_kwargs={}):

        if 'lam' not in model_params.keys():
            lam = self.dtype(1.0)
            print("Using default lam = 1.0")
        else:
            lam = self.dtype(model_params['lam'])

        if 'epsilon1' not in model_params.keys():
            epsilon1 = self.dtype(5e-3)
        else:
            epsilon1 = self.dtype(model_params['epsilon1'])

        if 'epsilon2' not in model_params.keys():
            epsilon2 = self.dtype(5e-2)
            print("Using default epsilon2 = 5e-2")
        else:
            epsilon2 = self.dtype(model_params['epsilon2'])

        if 'epsilon3' not in model_params.keys():
            epsilon3 = self.dtype(5e-3)
        else:
            epsilon3 = self.dtype(model_params['epsilon3'])

        if 'w' not in model_params.keys():
            print("Using default w = 1.0")
            w = self.dtype(1.0)
        else:
            w = self.dtype(model_params['w'])

        if 'r' not in model_params.keys():
            r = self.dtype(0.1)
            epsilon2 = r * epsilon2

        else:
            r = self.dtype(model_params['r'])
            epsilon2 = r * epsilon2

        if Y0 is None:
            Y0 = jnp.zeros(2*(self.T-1)*self.N + 3*(self.T * self.N) + self.T, dtype=self.dtype)

      
        t0 = time.time()
        S = solver.jaxSolver(lam=lam, 
                             epsilon1=epsilon1, 
                             epsilon2=epsilon2, 
                             epsilon3=epsilon3, 
                             w=w, r=r, C=self.C, g=self.all_growth, 
                             col_t=self.data_dists, T=self.T, N=self.N, ages=self.times)

        sol, ran_iter, error = S.solve(Y0=Y0, max_iter=max_iter, constraint_tol=constraint_tol, verbose=verbose, **solver_kwargs)
        tt = time.time() - t0

        cpu = jax.devices("cpu")[0]
        with jax.default_device(cpu):
            params = np.asarray(sol.params, dtype=np.float32) 
            pi_array = S.get_pi_from_Y(params)


        solver_vals = {'optimization time': f'{float(tt) / 60:.2f} mins',
                    'objective value': float(sol.state.value),
                    'grad_norm': float(sol.state.error),
                    'n_iter': int(ran_iter),
                    'failed line search': bool(sol.state.failed_linesearch),
                    'max constraint error': error}

        solver_df = pd.DataFrame(solver_vals, index=[0])
        self.solver_df = solver_df

        if self.save_dir is not None:
            solver_df.to_csv(os.path.join(self.save_dir, 'solver_results.csv'), index=False)

        else:
            print(f"Ran {ran_iter} iterations in {tt/60:.2f} minutes.")
            print(f"Final objective value: {sol.state.value:.4e}\n" + 
                  f"grad norm: {sol.state.error:.3e}\n" +
                  f"max constraint error: {error:.3e}\n" +
                  f"ended in failed line search: {sol.state.failed_linesearch}")

        for i, t in enumerate(self.times):
            self.adata.obsp[f'pi_{t}'] = np.asarray(pi_array[i]/ np.sum(pi_array[i])) 


    def get_lin_fate_probs(self, label_key, all_labels=None, lin_fp_error_tol=1e-2, HDR_cutoff=1e-3):

        if all_labels is None:
            all_labels = np.unique(self.adata.obs[label_key])
        self.adata.uns[f'{label_key}_fp_labels'] = all_labels

        labels = self.adata.obs[label_key].to_numpy()


        jax.config.update("jax_enable_x64", True)
        for i in range(self.T):
            growth_i = self.all_growth[i]
            sink_idx = growth_i < 1


            if sink_idx.sum() == 0:
                print(f"No sinks at time {self.times[i]}, cannot compute fate probabilities.")
                lin_fp = np.nan * np.ones((labels.shape[0], all_labels.shape[0]))
                self.adata.obsm[f'{label_key}_fp_t={self.times[i]}'] = lin_fp
                continue
            
            pi = self.adata.obsp[f'pi_{self.times[i]}']

            dist = pi.sum(0)

            if HDR_cutoff > 0:
                # Trim distribution to HDR: remove lowest probability states with collective mass < HDR_cutoff
                # i.e. for more stable computation of fate probabilities we restrict to states containing 99.9% of 
                # the probability mass
                dist_ordered, dist_idx = jnp.sort(dist), jnp.argsort(dist)
                cumsum_dist = jnp.cumsum(dist_ordered)
                cutoff_idx = jnp.searchsorted(cumsum_dist, HDR_cutoff)
                low_mass_idx = dist_idx[:cutoff_idx]
                non_zero_idx = jnp.ones(pi.shape[0], dtype=bool).at[low_mass_idx].set(False)
    #            non_zero_idx = np.asarray(mask)
            else:
                non_zero_idx = dist > 0

            pi_non_zero = pi[non_zero_idx][:, non_zero_idx]
            labels_non_zero = labels[non_zero_idx]
            sink_idx_non_zero = sink_idx[non_zero_idx]  

            P = utils.row_normalize(jnp.asarray(pi_non_zero, jnp.float64), sink_mask=sink_idx_non_zero)
            if not np.allclose(P.sum(1), 1.0):
                print(f"Warning: Transition matrix rows do not sum to 1 at time {self.times[i]}.")
                lin_fp = np.nan * np.ones((labels.shape[0], all_labels.shape[0]))
                self.adata.obsm[f'{label_key}_fp_t={self.times[i]}'] = lin_fp
                continue

            try:
                lin_fp, sink_labels = utils.compute_fate_probs_lineages(P, sink_idx_non_zero, labels_non_zero)

            except np.linalg.LinAlgError as e:
                print(f"Warning: Singular transition matrix; cannot compute fate probabilities: {e}")
                lin_fp = np.nan * np.ones((len(growth_i), len(all_labels)))
                self.adata.obsm[f'{label_key}_fp_t={self.times[i]}'] = lin_fp
                continue

            if len(lin_fp) == 0:
                print(f"Warning: No lineages found at time {self.times[i]}, setting fate probabilities to NaN.")
                lin_fp = np.nan * np.ones((len(growth_i), len(all_labels)))
                self.adata.obsm[f'{label_key}_fp_t={self.times[i]}'] = lin_fp
                continue

            if len(lin_fp.shape) < 2:
                lin_fp = lin_fp.reshape(-1, 1)

            if lin_fp.shape[0] != P.shape[0]:
                print(f"Warning: Fate prob shape {lin_fp.shape} does not match number of non-sink states {P.shape[0]}, setting to NaN.") 
                lin_fp = np.nan * np.ones((len(growth_i), len(all_labels)))
                self.adata.obsm[f'{label_key}_fp_t={self.times[i]}'] = lin_fp
                continue

            lin_fp_error = np.max(np.abs(lin_fp.sum(1) - 1.0))

            if (lin_fp_error > lin_fp_error_tol):
                print(f"Warning: Max lineage fate prob error {lin_fp_error:.5f} exceeds tolerance {lin_fp_error_tol:.5f} at time {self.times[i]}, setting to NaN.")
                lin_fp = np.nan * np.ones_like(lin_fp)

            else:
                lin_fp = np.asarray(utils.row_normalize(lin_fp), dtype=np.float32)
            
            # add a 0 column for cell types with no sinks
            fp_lin_all= np.zeros((lin_fp.shape[0], len(all_labels)), dtype=np.float32)
            for j, ct in enumerate(all_labels):
                if ct in sink_labels:
                    idx = np.where(sink_labels == ct)[0][0]
                    fp_lin_all[:, j] = lin_fp[:, idx]
            # add back in nans for 0 mass points
            fp_lin_all_full = np.zeros((labels.shape[0], len(all_labels)), dtype=np.float32)
            fp_lin_all_full[non_zero_idx] = fp_lin_all

            self.adata.obsm[f'{label_key}_fp_t={self.times[i]}'] = fp_lin_all_full

        if self.dtype != jnp.float64:
            jax.config.update("jax_enable_x64", False)

    def get_trajectories(self, num_step, num_traj, key='traj_data', plot_hitting_time=False, plot_traj=False):

        if type(num_step) is int:
            num_step_list = [num_step] * self.T
        if type(num_traj) is int:
            num_traj_list = [num_traj] * self.T

        transition_matrices = [utils.row_normalize(self.adata.obsp[f'pi_{self.times[i]}']) for i in range(self.T)]
        init_dists = [jnp.clip(self.adata.obs[self.growth_rate_key].to_numpy(), 0, None) *
                      self.adata.obsp[f'pi_{self.times[i]}'].sum(0) for i in range(self.T)]

        traj_data_ind = utils.get_traj_distributions(transition_matrices=transition_matrices,
                                                  init_dists=init_dists,
                                                  num_traj_list=num_traj_list,
                                                  num_step_list=num_step_list)

        test_traj_data = [self.adata.obsm[self.embed_key][traj_data_ind[i]] for i in range(len(traj_data_ind))]

        self.adata.uns[key] = {}

        for i, age in enumerate(self.times):
            self.adata.uns[key][str(age)] = test_traj_data[i]

        if plot_hitting_time:
            utils.plot_time_to_sink(self.times,
                                    n_points_list=[self.N for _ in range(self.T)],
                                    traj_data_ind=traj_data_ind,
                                    sink_idx_list=[np.where(self.all_growth_rates[i] < 0)[0] for i in range(self.T)]
            )

        if plot_traj:
            if self.save_dir is not None:
                utils.plot_trajectories(self.adata, N=50, ncols=5, imsize=5,
                                        sup_title=f"gStatOT Trajectories",
                                        save_path=os.path.join(self.save_dir, f'gStatOT_trajectories.png'))
            else:
                utils.plot_trajectories(self.adata, N=50, ncols=5, imsize=5,
                                        sup_title=f"gStatOT Trajectories",
                                        save_path=None)

       
