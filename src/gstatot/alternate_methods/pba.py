
import subprocess
import os
import numpy as np
from ..utilities import utils
import jax.numpy as jnp

# Wrapper for Python 2 script that runs PBA on each time point of the data and saves results in adata.obsm['cell_type_key' + '_fp'] 
# and adata.obs['mfpt'] (if run_mfpt is True). The, unaugmented and unormalized Markov transition matrix is saved in 
# adata.uns['pi_' + time] for each time point.

class PBA:

    def __init__(self, adata, adata_keys, script_path, save_dir, k=10, D=1.0, E=-10000, V=0,
                 growth_rate_func=None, py2_path=None, run_mfpt=False, del_save_dir=True, 
                 res_save_dir=None, verbose=False, lin_fp_error_tol=1e-2,
                 use_pca=True) -> None:

        self.adata = adata
        self.adata_key = adata_keys
        self.script_path = script_path
        self.k = k
        self.D = D
        self.adata_keys = adata_keys
        self.py2_path = py2_path if py2_path is not None else 'python2'
        self.run_mfpt = run_mfpt
        self.verbose = verbose
        self.lin_fp_error_tol = lin_fp_error_tol

        self.del_save_dir = del_save_dir
        self.save_dir = save_dir # Need to save intermediate files for Python 2 script; will be deleted after script runs if del_save_dir is True
        os.makedirs(save_dir, exist_ok=True) if save_dir is not None else None

        self.res_save_dir = res_save_dir # Dir to save figures
        os.makedirs(res_save_dir, exist_ok=True) if res_save_dir is not None else None

        adata_keys_needed = ['time_key', 'growth_rate_key']

        if not all([key in adata_keys.keys() for key in adata_keys_needed]):
            raise ValueError(f"adata_keys must contain the following keys: {adata_keys_needed}")

        for key in adata_keys.keys():
            setattr(self, key, adata_keys[key])

        self.times = np.unique(self.adata.obs[self.time_key].to_numpy().astype(np.float32))

        self.N = self.adata.shape[0]
        self.T = len(self.times)

        # growth rates
        if growth_rate_func is not None:
            self.all_growth_rates = [growth_rate_func(self.adata[self.adata.obs[self.time_key] == time].obsm[self.embed_key], time) 
                                               for time in self.times]
        else:
            self.all_growth_rates = [self.adata[self.adata.obs[self.time_key] == time].obs[self.growth_rate_key].to_numpy() for time in self.times]

        # default parameters for PBA script
        self.default_X_path = ''      # <path_to_expression_matrix>            (required if no edge list is supplied; .npy or .csv)  
        self.default_E_value = -10000 # <minimum_mean expression>              (default = -10000; used to filter genes)
        self.default_V_value = 0      # <minimum_CV>                           (default = 0; used to filter genes)
        self.default_N_value = False  # <Normalize>                            (default = False; used to normalize expression data for knn graph))
        self.default_p_value = 50     # <PCA dimension>                        (default = 50; used to compute distance matrix)
        self.default_k_value = self.k # <number of nearest neighbors>          (default = 10; used to compute edge list)
        self.default_e_path = ''      # <path_to_edge_list>                    (required if no expression matrix is supplied)
        self.default_R_path = ''      # <path_to_sources_sinks_vector>         (required; .npy or .csv)
        self.default_S_path = ''      # <path_to_lineage_specific_sink_matrix> (optional, needed to compute fate probabilities; .csv or .npy)
        self.default_D_value = self.D #                                        (default = 1.0; controls the level of stochasticity in the model)


        all_lineages = self.adata.uns['all_cell_types']                        

        self.adata.obsm[adata_keys['cell_type_key'] + '_fp'] = np.zeros((self.adata.shape[0], len(all_lineages)))

        if self.run_mfpt:
            self.adata.obs['mfpt'] = np.nan

        terminal_lineages = np.unique(self.adata[self.adata.obs[self.growth_rate_key] < 0].obs[self.adata_keys['cell_type_key']].values)

        for age in self.times:
            age_adata = self.adata[self.adata.obs[self.time_key] == age]
            if use_pca:
#                print("Using existing PCA embedding for PBA")
                X = age_adata.obsm[self.embed_key]
            else:
                X = age_adata.X

            X_path = os.path.join(self.save_dir, f'X_{age}.npy')
            np.save(X_path, X)

            growth_rates = age_adata.obs[self.growth_rate_key].to_numpy()
            growth_rate_path = os.path.join(self.save_dir, f'growth_rates_{age}.npy')
            np.save(growth_rate_path, growth_rates)

            # get sink matrix
            sink_matrix = np.zeros((age_adata.shape[0], len(terminal_lineages)))
            sink_mask = age_adata.obs[self.adata_keys['growth_rate_key']].values < 0 
            for num, lineage in enumerate(terminal_lineages):
                mask = sink_mask & (age_adata.obs[self.adata_keys['cell_type_key']].values == lineage)
                sink_matrix[mask, num] = - age_adata.obs[self.adata_keys['growth_rate_key']].values[mask]

            assert np.all(sink_matrix >= 0)

            S_path = os.path.join(self.save_dir, f'sink_matrix_{age}.npy')
            np.save(S_path, sink_matrix)

            kwargs = {
                'X': X_path,
                'R': growth_rate_path,
                'E': self.default_E_value,
                'V': self.default_V_value,
                'N': self.default_N_value,
                'p': self.default_p_value,
                'k': self.default_k_value,
                'D': self.default_D_value,
                'S': S_path,
                'P': self.py2_path,
                'W': self.script_path,
                'M': self.run_mfpt
            }
            # make script path absolute
            kwargs['W'] = os.path.abspath(kwargs['W'])
            kwargs['P'] = os.path.abspath(kwargs['P'])
            kwargs['X'] = os.path.abspath(kwargs['X'])
            kwargs['R'] = os.path.abspath(kwargs['R'])
            kwargs['S'] = os.path.abspath(kwargs['S'])
            self.script_path = os.path.abspath(self.script_path)
            self.save_dir = os.path.abspath(self.save_dir)

            self.run_python2_script(self.script_path, **kwargs)

            fate_probs = np.load(os.path.join(self.save_dir, f'B.npy'))
            lin_fp_error = np.max(np.abs(fate_probs.sum(1) - 1))

            if lin_fp_error > self.lin_fp_error_tol: 
                print(f"Warning: {lin_fp_error:.5e} lineage fate probability error above tolerance of {self.lin_fp_error_tol}. Setting fate probabilities to NaN for age {age}.")
                fate_probs = np.nan * np.ones_like(fate_probs)

            if self.run_mfpt:
                mfpt = np.load(os.path.join(self.save_dir, f'T.npy'))
                # want mfpt from source to each cell
                source_mask = age_adata.obs[self.adata_keys['growth_rate_key']].values > 0
                source_mfpt = mfpt[source_mask, :].sum(axis=0)/ source_mask.sum() # average mfpt from sources to each cell
                self.adata.obs.loc[age_adata.obs_names, 'mfpt']= source_mfpt

            pi = np.load(os.path.join(self.save_dir, f'P.npy'))


            # get indices of terminal lineages in all_lineages, then add zero columns to fate_probs for any missing terminal lineages, and reorder columns to match order of all_lineages
            fate_probs_full = np.zeros((fate_probs.shape[0], len(all_lineages)))

            for i, lineage in enumerate(terminal_lineages):
                lineage_idx = np.where(all_lineages == lineage)[0][0]
                fate_probs_full[:, lineage_idx] = fate_probs[:, i]

            self.adata.obsm[adata_keys['cell_type_key'] + '_fp'][self.adata.obs[self.time_key] == age, :] = fate_probs_full

            for lineage in terminal_lineages:
                self.adata.obs[f'PBA_{lineage}_fp'] = self.adata.obsm[adata_keys['cell_type_key'] + '_fp'][:, all_lineages == lineage]

            self.adata.uns[f'pi_{age}'] = pi

            # delete intermediate files
            if self.del_save_dir:
                for file in os.listdir(self.save_dir):
                    os.remove(os.path.join(self.save_dir, file))

        if self.del_save_dir:
            os.rmdir(self.save_dir)

    def run_python2_script(self, script_path, **kwargs):
        # Resolve the absolute path to the pyenv Python 2 executable
        py2_executable = kwargs.get('P', 'python2')  # Default to 'python2' if not provided 
        
        # Build the command using the absolute path
        cmd = [py2_executable, script_path + '/PBA_pipeline.py']
        for key, value in kwargs.items():
            cmd.append(f'-{key}')
            cmd.append(str(value))

        if not self.verbose:
            sout = subprocess.DEVNULL
            serr = subprocess.DEVNULL
        else:
            sout = None
            serr = None
        
        try:
            result = subprocess.check_call(
                cmd,
                stdout=sout,
                stderr=serr,
                cwd=script_path
            )
            
        except subprocess.CalledProcessError as e:
            print(f"Python 2 Script Failed with error:\n{e.stderr}")


    def get_trajectories(self, num_step, num_traj, key='PBA_traj_data', plot_hitting_time=False, plot_traj=False, return_indices=False, return_traj_len=False):

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

        if plot_hitting_time:
            traj_len = utils.plot_time_to_sink(self.times,
                                    n_points_list=[len(self.all_growth_rates[i]) for i in range(self.T)],
                                    traj_data_ind=traj_data_ind,
                                    sink_idx_list=[np.where(self.all_growth_rates[i] < 0)[0] for i in range(self.T)],
                                    return_traj_len=True)

        if plot_traj:
            if self.res_save_dir is not None:
                utils.plot_trajectories(f'PBA', self.adata, N=np.minimum(num_traj, 50), ncols=5, imsize=5,
                                        sup_title=f"PBA Trajectories, k = {self.k}, D = {self.D}, Steps = {num_step}",
                                        save_path=os.path.join(self.res_save_dir, f'PBA_trajectories.png'))
            else:
                utils.plot_trajectories(f'PBA', self.adata, N=np.minimum(num_traj, 50), ncols=5, imsize=5, 
                                        sup_title=f"PBA Trajectories - k = {self.k}, D = {self.D}, Steps = {num_step}", 
                                        save_path=None)

        if return_indices:
            return traj_data_ind

        elif return_traj_len:
            return traj_len