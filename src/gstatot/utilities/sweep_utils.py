import gc
import os
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from . import utils
from .. import Metric_Evaluator 
from .. import gStatOT
from gstatot.alternate_methods import StatOT
from gstatot.alternate_methods.pba import PBA
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import scanpy as sc

def fit_method(method_str, method_params, adata, adata_keys, comp_fp=True, comp_traj=True):

    if method_str == 'gStatOT':

        model = gStatOT(adata=adata, adata_keys=adata_keys, dt=method_params['dt'], dtype=method_params.get('dtype', jnp.float32))

        fit_params = {'lam': method_params['lam'],
                      'w' : method_params.get('w', 1.0),
                      'epsilon2': method_params['epsilon2'],
                      'epsilon1': method_params.get('epsilon1', 0.005),
                      'epsilon3': method_params.get('epsilon3', 0.005),
                      'r': method_params.get('r', 1.0)}

        model.fit(model_params=fit_params, 
                  max_iter=method_params.get('max_iter', 1000), 
                  tol=method_params.get('tol', 1e-3),
                  verbose=False,
                  solver_type=method_params.get('solver_type', 'BCA'))
        
        if comp_fp:
            model.get_lin_fate_probs(label_key=adata_keys['cell_type_key'],
                                     all_labels=adata.uns['all_cell_types'])

        if comp_traj:

            model.get_trajectories(num_step=method_params['num_step'],
                                  num_traj=method_params['num_traj'],
                                  plot_hitting_time=False,
                                  plot_traj=False)
        
        fit_adata = model.adata
        fit_adata.uns['model_dt'] = method_params['dt']
        full_supp = True
        solver_df = model.solver_df

        

    elif method_str == 'StatOT':
        
        model = StatOT(adata=adata, adata_keys=adata_keys, dt=method_params['dt'], 
                       dtype=method_params.get('dtype', jnp.float32))

        fit_params = {'epsilon': method_params['epsilon'],
                      'lse': True,
                      'cost_scaling': 'mean'}
        
        model.fit(model_params=fit_params, 
                  max_iter=method_params.get('max_iter', 100_000),
                  constraint_tol=method_params.get('constraint_tol', 1e-6),
                  verbose=False,
                  )
        
        if comp_fp:
            model.get_lin_fate_probs(label_key=adata_keys['cell_type_key'], all_labels=adata.uns['all_cell_types'])

        if comp_traj:
            model.get_trajectories(num_step=method_params['num_step'], num_traj=method_params['num_traj'], plot_hitting_time=False, plot_traj=False)

        fit_adata = model.adata
        fit_adata.uns['model_dt'] = method_params['dt']
        full_supp = False 
        solver_df = model.solver_df

    elif method_str == 'PBA':

        k = method_params['k']
        D = method_params['D']
        save_dir = method_params['save_dir']
        use_pca = method_params.get('use_pca', False)
        script_path = method_params['script_path']
        py2_path = method_params.get('py2_path', None)

        model = PBA(adata=adata, adata_keys=adata_keys, script_path=script_path, save_dir=save_dir, k=k, D=D,
                    py2_path=py2_path, use_pca=use_pca)

        if comp_traj:
            model.get_trajectories(num_step=method_params['num_step'], num_traj=method_params['num_traj'], plot_traj=False)

        fit_adata = model.adata
        fit_adata.uns['model_dt'] = method_params['dt'] # for trajectory error calculation
        full_supp = False
        solver_df = pd.DataFrame({'n_iter': [np.nan], 'optimization time': [np.nan], 'const_err': [np.nan]})

    else:
        raise ValueError("Method string not recognized. Must be one of 'gStatOT', 'StatOT', or 'PBA'.")
    
    return fit_adata, full_supp, solver_df

def eval_method(method, test_adata, true_adata, full_supp, adata_keys, comp_fp=True, comp_traj=True, comp_prop=True):

    metric_tests = Metric_Evaluator(method=method,
                                    test_adata=test_adata,
                                    true_adata=true_adata,
                                    time_key=adata_keys['time_key'],
                                    embed_key=adata_keys['embed_key'],
                                    plot_metrics=False,
                                    full_supp=full_supp)

    metric_tests.w2_marginal_error()

    if comp_fp:
        metric_tests.fp_tv_error(label_key=adata_keys['cell_type_key'])
 
    if comp_traj:
        metric_tests.w2_trajectory_error(test_dt=test_adata.uns['model_dt'], true_dt=true_adata.uns['true_dt'])

    if comp_prop:
        metric_tests.prop_TV_error(type_key=adata_keys['cell_type_key'])

    return metric_tests.results_df



def sweep_method(method, file_name, true_adata, adata_keys, method_params_list, num_exp=None, T_time=None, n=None, 
                 save_dir=None, seed=42, fit_true_adata=False, metric_weights=None, comp_fp=True, comp_traj=True, comp_prop=True,
                 ):
    '''Helper function to run a single method across all resampling experiments for a given T and n.
       example method_params_dict = {1: {'dt': dt, 'dtype': dtype, 'epsilon': epsilon}, 2: {...}, ...}'''

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    all_res_df = pd.DataFrame()

    v_sets = False

    if type(true_adata) == list:

        v_sets = True
        validations_sets = true_adata
        num_exp = len(true_adata)

    num_metrics = np.sum([comp_fp, comp_traj, comp_prop]) + 1 # marginal W2 is always computed, add 1 for each additional metric
    if metric_weights is None:
        metric_weights = 1/num_metrics * np.array([1] + [comp_fp, comp_traj, comp_prop], dtype=np.float32) # default is equal weighting of all metrics, with 0 weight for metrics that aren't computed

    if T_time is None:
        T_time = jnp.unique(jnp.asarray(true_adata.obs[adata_keys['time_key']], dtype=np.float32))

    N = len(T_time) * n

    key = jax.random.PRNGKey(seed)

    for exp in range(1, num_exp + 1):
    # downsample adata to n cells per time point

        if v_sets:
            t_adata = sc.read_h5ad(validations_sets[exp - 1])

        else:
            t_adata = true_adata

        key, subkey = jax.random.split(key)
        downsampled_adata = utils.downsample_adata_by_age(t_adata, n=n, PRNG_KEY=subkey, 
                                                          chosen_times=T_time, 
                                                          time_key=adata_keys['time_key'])

        print(f"\n Experiment {exp}/{num_exp} - Fitting method: {method} ---------")
        for params in tqdm(method_params_list):

            fit_adata, full_supp, solver_df = fit_method(method_str=method, method_params=params, adata=downsampled_adata, adata_keys=adata_keys, comp_fp=comp_fp, comp_traj=comp_traj)

            _results_df = eval_method(method=method, test_adata=fit_adata, true_adata=t_adata, 
                                      full_supp=full_supp, adata_keys=adata_keys,
                                      comp_fp=comp_fp,
                                      comp_traj=comp_traj,
                                      comp_prop=comp_prop)

            # add solver_df to _results_df
            for col in solver_df.columns:
                _results_df[col] = solver_df[col].values[0]
            # add params to results_df
            for param_key, param_val in params.items():
                _results_df[param_key] = param_val

            _results_df['n'] = n
            _results_df['T'] = len(T_time)
            _results_df['N'] = N
            _results_df['experiment'] = exp

            all_res_df = pd.concat([all_res_df, _results_df], ignore_index=True)

            # add column to all_res_df that is true if FP_TV_dist is nan
            all_res_df['FP_TV_dist_is_nan'] = all_res_df['FP_TV_dist'].isna()
            all_res_df.loc[all_res_df['FP_TV_dist_is_nan'], 'FP_TV_dist'] = 1.0 # set FP_TV_dist to 1 for methods that don't predict fate probabilities
        
            all_res_df['scaled_marginal_W2_dist'] = (all_res_df['marginal_W2_dist'] - np.nanmin(all_res_df['marginal_W2_dist'])) / (np.nanmax(all_res_df['marginal_W2_dist']) - np.nanmin(all_res_df['marginal_W2_dist']))

            if comp_fp:
                all_res_df['scaled_FP_TV_dist'] = (all_res_df['FP_TV_dist'] - np.nanmin(all_res_df['FP_TV_dist'])) / (np.nanmax(all_res_df['FP_TV_dist']) - np.nanmin(all_res_df['FP_TV_dist']))
            else:
                all_res_df['scaled_FP_TV_dist'] = 0.0 # set scaled_FP_TV_dist to 0 for methods that don't predict fate probabilities

            if comp_traj:
                all_res_df['scaled_traj_W2_dist'] = (all_res_df['traj_W2_dist'] - np.nanmin(all_res_df['traj_W2_dist'])) / (np.nanmax(all_res_df['traj_W2_dist']) - np.nanmin(all_res_df['traj_W2_dist']))

            else:
                all_res_df['scaled_traj_W2_dist'] = 0.0 # set scaled_traj_W2_dist to 0 for methods that don't predict trajectories
        
            if comp_prop:
                all_res_df['scaled_prop_TV_dist'] = (all_res_df['prop_TV_dist'] - np.nanmin(all_res_df['prop_TV_dist'])) / (np.nanmax(all_res_df['prop_TV_dist']) - np.nanmin(all_res_df['prop_TV_dist']))

            else:
                all_res_df['scaled_prop_TV_dist'] = 0.0 # set scaled_prop_TV_dist to 0 for methods that don't predict proportions

       
            all_res_df['mean_metric'] = metric_weights[0] * all_res_df['scaled_marginal_W2_dist'] + metric_weights[1] * all_res_df['scaled_traj_W2_dist'] \
                                      + metric_weights[2] * all_res_df['scaled_FP_TV_dist'] + metric_weights[3] * all_res_df['scaled_prop_TV_dist']

            if save_dir is not None:
                all_res_df.to_csv(save_dir + f'/{method}_{file_name}_n={n}_T={len(T_time)}_sweep_results.csv')

            gc.collect()
            jax.clear_caches()

        # free up memory
        del downsampled_adata 
        del fit_adata
        del _results_df
        gc.collect()
        jax.clear_caches()



    return all_res_df

def plot_sweep_results(results_df, param1, param2, metric='mean_metric', save_dir=None, input_ax=None, 
                       vminmax=(None, None), title=None, cbar=True, highlight_nans=False, cmap='YlGnBu'):

    if input_ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        ax = input_ax
    
    if param2 is not None:
        pivot_table = results_df.pivot_table(values=metric, index=param1, columns=param2, aggfunc='mean', fill_value=np.nan, dropna=False)
    else:
        pivot_table = results_df.groupby(param1)[metric].mean().reset_index()
        pivot_table = pivot_table.set_index(param1)
    
    vmin, vmax = vminmax
    sns.heatmap(pivot_table, annot=False, fmt=".2f", cmap=cmap, vmin=vmin, vmax=vmax, ax=ax, cbar=cbar)
    # add red patch around min value
    opt_param1, opt_param2 = None, None
    min_val = pivot_table.min().min()
    for i in range(pivot_table.shape[0]):
        for j in range(pivot_table.shape[1]):
            if pivot_table.iloc[i, j] == min_val:
                opt_param1 = pivot_table.index[i]
                opt_param2 = pivot_table.columns[j]
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=3, alpha=0.5))
                # change font color based on background color
                ax.text(j + 0.5, i + 0.5, f'{pivot_table.iloc[i, j]:.2f}', 
                        ha='center', va='center', color='black', fontsize=14)
    if highlight_nans:
        nan_pivot = results_df.pivot_table(values='FP_test_nans', index=param1, columns=param2, aggfunc='mean', fill_value=0, dropna=False)
        for i in range(nan_pivot.shape[0]):
            for j in range(nan_pivot.shape[1]):
                if nan_pivot.iloc[i, j] > 0:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="#FFA600", lw=4))

    title_str = metric if title is None else title
    ax.set_title(title_str)
    if save_dir is not None:
        plt.savefig(save_dir + f'/{metric}_by_{param1}_and_{param2}.png')
        plt.close()
        return opt_param1, opt_param2
    else:
        if input_ax is not None:
            return opt_param1, opt_param2

        else:
            plt.show()
            plt.close()
            return opt_param1, opt_param2
