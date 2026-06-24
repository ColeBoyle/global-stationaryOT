import time
from . import solver
import jax
import jax.numpy as jnp
import numpy as np
import gstatot.utilities.utils as utils
import pandas as pd
import os


class gStatOT:

    def __init__(self, adata, adata_keys, dt=0.25, cost_scaling='mean', 
                 growth_rate_func=None, dtype=jnp.float32, save_dir=None) -> None:
        """Initialize the gStatOT model.

        :param adata: AnnData object containing the single-cell data.
        :type adata: anndata.AnnData
        :param adata_keys: Dictionary containing the keys for time, embedding, and growth rate
        :type adata_keys: dict
        :param dt: Estimated time period for cell state transitions. Default is 0.25.
        :type dt: float
        :param cost_scaling: Method for scaling the cost matrix. Default is 'mean'. Other options are 'max', 'median', or a float value.
        :type cost_scaling: str
        :param growth_rate_func: Function, R(x,a), mapping cell states and age to growth rates. 
                                 If None, growth rates are assumed to be in the obs of the AnnData object. Default is None.
        :type growth_rate_func: callable
        :param dtype: Data type for computations. Either jax.numpy.float32 or jax.numpy.float64. Default is jax.numpy.float32.
        :type dtype: jax.numpy.dtype
        :param save_dir: Directory to save results. Default is None.
        :type save_dir: str
        """

        if dtype == jnp.float64:
            jax.config.update("jax_enable_x64", True)
        else:
            jax.config.update("jax_enable_x64", False)

        self.dtype = dtype

        self.adata = adata
        self.dt = self.dtype(dt) 
        self.cost_scaling = cost_scaling.capitalize() if isinstance(cost_scaling, str) else cost_scaling

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True) if save_dir is not None else None

        adata_keys_needed = ['time_key', 'embed_key']

        if not all([key in adata_keys.keys() for key in adata_keys_needed]):
            raise ValueError(f"adata_keys must contain the following keys: {adata_keys_needed}")

        for key in adata_keys.keys():
            setattr(self, key, adata_keys[key])

        self.times = jnp.unique(jnp.asarray(self.adata.obs[self.time_key], dtype=self.dtype))
        X = jnp.asarray(self.adata.obsm[self.embed_key], dtype=self.dtype)

        self.N = X.shape[0]
        self.T = len(self.times)
        self.C = utils.vmap_sq_dist(X, X)

        if self.cost_scaling == 'Max':
            self.C = self.C / jnp.max(self.C)

        elif self.cost_scaling == 'Mean':
            self.C = self.C / jnp.mean(self.C)

        elif self.cost_scaling == 'Median':
            self.C = self.C / jnp.median(self.C)

        elif isinstance(self.cost_scaling, float):
            self.C = self.C / self.dtype(self.cost_scaling)

        elif (self.cost_scaling != None):
            raise ValueError("cost_scaling must be float, 'Max', 'Mean' or 'Median'")

        # growth rates
        if growth_rate_func is not None:
            self.all_growth_rates = jnp.array([growth_rate_func(X, time) for time in self.times], dtype=self.dtype)
        else:
            # check if growth_rate_key is in adata.obs
            if 'growth_rate_key' not in adata_keys.keys():
                raise ValueError("If growth_rate_func is None, adata_keys must contain 'growth_rate_key'")

            self.growth_rates = jnp.array(self.adata.obs[self.growth_rate_key], dtype=self.dtype)
            self.all_growth_rates = jnp.array([self.growth_rates for _ in range(self.T)], dtype=self.dtype)

        self.all_growth = jnp.exp(self.all_growth_rates * self.dt)

        data_dists = []
        for t in self.times:
            col_cur = np.zeros(self.N)
            col_cur[self.adata.obs[self.time_key] == t] = 1.0
            col_cur = col_cur / np.sum(col_cur)
            data_dists.append(col_cur)

        #if penalize_self_transitions:
        #   # penalize transitions non growth states 
        #   self.C = jnp.where(self.all_growth_rates == 0, self.C + jnp.eye(self.N) * jnp.max(self.C), self.C)
        #   print("Penalizing self-transitions by adding max cost to diagonal of cost matrix.")
        

        self.data_dists = jnp.array(data_dists, dtype=self.dtype)

    def set_save_dir(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True) if save_dir is not None else None

    def fit(self, model_params={}, max_iter=1000, Y0=None, verbose=False, return_params=False,
            tol=1e-3, solver_kwargs={}, solver_type='BCA'):
        """Fit the gStatOT model to the time course.
        
        :param model_params: Dictionary containing the model parameters 'lam', 'epsilon1', 'epsilon2', 'epsilon3', 'w'. 
        If a parameter is not provided, the following default values will be used. lam = 1.0, epsilon1 = 5e-3, epsilon2 = 2.5e-2, epsilon3 = 5e-3, w = 1.0
        :type model_params: dict
        :param max_iter: Maximum number of (outer) iterations for the solver. Default is 1000.
        :type max_iter: int
        :param Y0: Initial guess for the dual variables. If None, a default random initialization is used. Default is None.
        :type Y0: jax.numpy.ndarray
        :param verbose: If True, print solve time, duality gap, and constraint error after the solver finishes.
        :type verbose: bool
        :param return_params: If True, return the optimized dual variables after fitting. Default is False.
        :type return_params: bool
        :param tol: Relative duality gap tolerance for the solver. Default is 1e-3.
        :type tol: float
        :param solver_kwargs: Additional keyword arguments to pass to the solver. Default is an empty dictionary.
        :type solver_kwargs: dict
        :param solver_type: Type of solver to use. Either 'BCA' or 'LBFGS', Default is 'BCA'.
        :type solver_type: str
        """

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
            epsilon2 = self.dtype(2.5e-2)
            if verbose:
                print("Using default epsilon2 = 2.5e-2")
        else:
            epsilon2 = self.dtype(model_params['epsilon2'])

        if 'epsilon3' not in model_params.keys():
            epsilon3 = self.dtype(5e-3)
        else:
            epsilon3 = self.dtype(model_params['epsilon3'])

        if 'w' not in model_params.keys():
            w = self.dtype(1.0)
        else:
            w = self.dtype(model_params['w'])

        if 'r' not in model_params.keys():
            r = self.dtype(1.0)
            epsilon2 = r * epsilon2
        else:
            r = self.dtype(model_params['r'])
            epsilon2 = r * epsilon2

        # check if we have any sinks in the data
        sink_mask = self.all_growth_rates < 0
        source_mask = self.all_growth_rates > 0
        if (not sink_mask.any() and source_mask.any()) or (not source_mask.any() and sink_mask.any()):
            print("Warning: Feasibility condition not met: returning NaN arrays")
            pi_array = np.full((self.T, self.N, self.N), np.nan, dtype=np.float32)
            solver_vals = {'optimization time': 'NaN',
                        'duality gap': np.nan,
                        'n_iter': np.nan,
                        'max constraint error': np.nan}
            solver_df = pd.DataFrame(solver_vals, index=[0])
    
        else:
            S = solver.jaxSolver(lam=lam, 
                                 epsilon1=epsilon1, 
                                 epsilon2=epsilon2, 
                                 epsilon3=epsilon3, 
                                 w=w, r=r, C=self.C, g=self.all_growth, 
                                 col_t=self.data_dists, T=self.T, N=self.N, ages=self.times,
                                 solver_type=solver_type, objective='W2')

            t0 = time.time()
            params, pi_array, gap, ran_iter, error = S.solve(Y0=Y0, max_iter=max_iter, 
                                           tol=tol, 
                                           verbose=verbose, 
                                           **solver_kwargs)
            tt = time.time() - t0
            ran_iter = ran_iter * solver_kwargs.get('inner_iter', 100) if solver_type == 'BCA' else ran_iter

            cpu = jax.devices("cpu")[0]
            with jax.default_device(cpu):
                params = np.asarray(params, dtype=np.float32) 
                pi_array = np.asarray(pi_array, dtype=np.float32)


            solver_vals = {'optimization time': f'{float(tt) / 60:.2f} mins',
                        'duality gap': float(gap),
                        'n_iter': int(ran_iter),
                        'max constraint error': error}

            solver_df = pd.DataFrame(solver_vals, index=[0])

            if verbose:
                print(f"Ran {ran_iter} iterations in {tt/60:.2f} minutes.")
                print(f"duality gap: {gap:.3e}\n" +
                      f"max constraint error: {error:.3e}\n")


        self.solver_df = solver_df
        if self.save_dir is not None:
            solver_df.to_csv(os.path.join(self.save_dir, 'solver_results.csv'), index=False)


        for i, t in enumerate(self.times):
            self.adata.obsp[f'pi_{t}'] = np.asarray(pi_array[i]/ np.sum(pi_array[i])) 

        if return_params:
            return params


    def get_lin_fate_probs(self, label_key, all_labels=None, 
                           lin_fp_error_tol=1e-2, 
                           init_HDT_cutoff=0.00, 
                           num_restarts=10):
        '''Compute cell lineage fate probabilities using inferred transition matrices and growth rates.

        :param label_key: Key in adata.obs containing cell lineage labels for fate computation. 
        :type label_key: str
        :param all_labels: List of all possible lineage labels. If None, unique labels from adata.obs[label_key] are used.
        :type all_labels: list or None
        :param lin_fp_error_tol: Tolerance for lineage fate probability computation. Returns NaN if error exceeds this value. Default is 1e-2.
        :type lin_fp_error_tol: float
        :param init_HDT_cutoff: Initial HDT bottom percentile cutoff for fate computation. Default is 0.00.
        :type init_HDT_cutoff: float
        :param num_restarts: Number of restarts for fate computation. Default is 10.
        :type num_restarts: int
        '''
        
        utils.get_lin_fate_probs(self, label_key=label_key, all_labels=all_labels, 
                                 lin_fp_error_tol=lin_fp_error_tol,
                                 full_supp=True, init_HDT_cutoff=init_HDT_cutoff, 
                                 num_restarts=num_restarts)

    def get_trajectories(self, num_step, num_traj, key='gStatOT_traj_data', plot_hitting_time=False, plot_traj=False, 
                         make_absorbing=False, make_transient=False, plotting_embed='X_pca', ncols=5, init_dist=None):
        """Sample cell trajectories from the inferred transition matrices.

        :param num_step: Number of steps to sample for each trajectory. Can be an int or a list of ints for each time point.
        :type num_step: int or list of int
        :param num_traj: Number of trajectories to sample. Can be an int or a list of ints for each time point.
        :type num_traj: int or list of int
        :param key: Key in adata.uns to store the sampled trajectory data. Default is 'gStatOT_traj_data'.
        :type key: str
        :param plot_hitting_time: If True, plot the time to hit sink states for the sampled trajectories. Default is False.
        :type plot_hitting_time: bool
        :param plot_traj: If True, plot the sampled trajectories. Default is False.
        :type plot_traj: bool
        :param make_absorbing: If True, make sink states absorbing in the transition matrices. Default is False.
        :type make_absorbing: bool
        :param make_transient: If True, set the the self transition probabilities for non-sink states to 0. Default is False.
        :type make_transient: bool
        :param plotting_embed: Key in adata.obsm to use for plotting trajectories. Default is 'X_pca'.
        :type plotting_embed: str
        :param ncols: Number of columns for trajectory plots. Default is 5.
        :type ncols: int
        :param init_dist: Initial distribution for sampling trajectories. If None, the initial distribution is computed from the cell growth 
        rates. Default is None.
        :type init_dist: jax.numpy.ndarray or None
        """

        if type(num_step) is int:
            num_step_list = [num_step] * self.T
        else:
            num_step_list = num_step
        if type(num_traj) is int:
            num_traj_list = [num_traj] * self.T
        else:
            num_traj_list = num_traj

        def TM(pi):
            pi = jnp.asarray(pi, dtype=self.dtype)
            if make_absorbing:
                growth_mask = self.growth_rates < 0
                pi = utils.row_normalize(pi, growth_mask, make_transient=make_transient)
            return pi

        transition_matrices = [TM(self.adata.obsp[f'pi_{self.times[i]}']) for i in range(self.T)]

        if init_dist is not None:
            init_dists = [jnp.asarray(init_dist, dtype=self.dtype) for _ in range(self.T)]

        else:
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
                utils.plot_trajectories('gStatOT', self.adata, N=np.minimum(num_traj, 50), ncols=ncols, imsize=5,
                                        sup_title=f"gStatOT Trajectories", traj_data_ind=traj_data_ind, embed_key=plotting_embed,
                                        save_path=os.path.join(self.save_dir, f'gStatOT_trajectories.png'))
            else:
                utils.plot_trajectories('gStatOT', self.adata, N=np.minimum(num_traj, 50), ncols=ncols, imsize=5,
                                        sup_title=f"gStatOT Trajectories", traj_data_ind=traj_data_ind, embed_key=plotting_embed,
                                        save_path=None)

    def get_mfpt(self):
        utils.get_mfpt(self, full_supp=True, dt=self.dt) 
