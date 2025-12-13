import jax
import jax.numpy as jnp
from jax import vmap, jit
from jax import random
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from gstatot import metrics, gStatOT, StatOT
import gc

sns.set_context("paper", font_scale=1.5)
sns.set(style="ticks")

def row_normalize(matrix, sink_mask=None, make_transient=False):
    # matrix: numpy array of shape (N, N)
    # sink_mask: numpy array of shape (N,) containing True for sink cells and False otherwise 

    if sink_mask is not None:
        matrix = matrix.at[sink_mask,:].set(0)
        matrix = matrix.at[sink_mask, sink_mask].set(1)
    # make non sink cells transient
    if make_transient and (sink_mask is not None):
        matrix = matrix.at[~sink_mask, ~sink_mask].set(0)

    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    return matrix

@jit
def sq_dist(x, z):
    return jnp.dot(x-z, x-z)

@jit
def l2_dist(x, y, z, w):
    return jnp.dot(x-z, x-z) +  jnp.dot(y-w, y-w)

@jit
def vmap_sq_dist(X, Z):
    return jax.vmap(jax.vmap(sq_dist, in_axes=(0, None)), in_axes=(None, 0))(X, Z)

@jit
def vmap_sq_dist_4(X, Y, Z, W):
    T = jax.vmap(jax.vmap(jax.vmap(jax.vmap(
        l2_dist, 
        in_axes=(0, None, None, None)), 
        in_axes=(None, 0, None, None)), 
        in_axes=(None, None, 0, None)), 
        in_axes=(None, None, None, 0))(X, Y, Z, W)

    return  T.reshape(X.shape[0] * Y.shape[0], Z.shape[0] * W.shape[0])

@jit
def vector_kernel(X, Y, Z, W, epsilon1):
    D = vmap_sq_dist_4(X, Y, Z, W)
    return jnp.exp(-D / epsilon1)

@jit
def point_kernel(C, epsilon2): 
    return jnp.exp(-  C /  epsilon2)

@jit
def direct_sum(a, b):
    return a[:, None] + b

direct_sum_vmap = jax.jit(vmap(direct_sum, in_axes=(0, 0), out_axes=0))

# taken from original StatOT implementation: https://github.com/zsteve/StationaryOT
def _compute_NS(P, sink_idx):
    Q = P[np.ix_(~sink_idx, ~sink_idx)]
    S = P[np.ix_(~sink_idx, sink_idx)]
    N = np.eye(Q.shape[0]) - Q
    return N, S

def compute_fate_probs(P, sink_idx):
    """Compute fate probabilities by individual sink cell

    :param P: transition matrix
    :param sink_idx: boolean array of length `N`, set to `True` for sinks and `False` otherwise.
    :return: matrix with dimensions `(N, S)` where `S` is the number of sink cells present.
    """
    N, S = _compute_NS(P, sink_idx)
    B = np.zeros((P.shape[0], sink_idx.sum()))
    B[~sink_idx, :] = np.linalg.solve(N, S)
    B[sink_idx, :] = np.eye(sink_idx.sum())
    return B

def compute_fate_probs_lineages(P, sink_idx, labels):
    """Compute fate probabilities by lineage

    :param P: transition matrix
    :param sink_idx: boolean array of length `N`, set to `True` for sinks and `False` otherwise.
    :param labels: string array of length `N` containing lineage names. Only those entries corresponding to sinks will be used.
    :return: matrix with dimensions `(N, L)` where `L` is the number of lineages with sinks.
    """
    B = compute_fate_probs(P, sink_idx)
    sink_labels = np.unique(labels[sink_idx])
    B_lineages = np.array([B[:, labels[sink_idx] == i].sum(1) for i in sink_labels]).T
    return B_lineages, sink_labels

def check_inf_norm_agreement(gamma, mu, nu):
    return max(jnp.linalg.norm(gamma.sum(1) - mu, ord = jnp.inf), jnp.linalg.norm(gamma.sum(0) - nu, ord = jnp.inf))
#### 

def plot_trajectories(adata, N, ncols=5, imsize=2, prefix="", sup_title=None, save_path=None, a=0.8, embed_key='X_pca'):
    traj_data = adata.uns['traj_data']
    ages = list(adata.uns['traj_data'].keys())
    traj_data = [adata.uns['traj_data'][age] for age in ages]
    point_data = adata.obsm[embed_key]

    # get N random trajectories
    randind = np.random.choice(jnp.arange(len(traj_data[0])), N, replace=False)
    n_rows = int(len(ages)/ncols) 
    n_rows = n_rows + 1 if len(ages) % ncols != 0 else n_rows

    fig, axes = plt.subplots(n_rows, ncols, sharex=True, sharey=True, figsize=(ncols * imsize, imsize * n_rows))
    axes = axes.flatten()

    if sup_title is None:
        fig.suptitle(prefix + "Trajectories")
    else:
        fig.suptitle(sup_title)

    for i, traj_dist in enumerate(traj_data):
        axes[i].plot(point_data[:,0], point_data[:,1], 'o', markersize=2, alpha=0.1, color='grey')

        for traj in traj_dist[randind]:
            axes[i].plot(traj[:,0], traj[:,1], 'o-', markersize=2, alpha=a)

        axes[i].set_title("Age " + str(ages[i]))

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    else:
        plt.show()
    plt.close()


def get_traj_distributions(transition_matrices, 
                           num_step_list, 
                           num_traj_list, 
                           init_dists, 
                           key=random.PRNGKey(0)):

    traj_data_ind = []

    if type(key) is int:
        key = random.PRNGKey(key)

    for i in range(len(transition_matrices)):

        max_num_step = num_step_list[i]
        num_traj = num_traj_list[i]

        P = jnp.asarray(transition_matrices[i])
        init_dist = jnp.asarray(init_dists[i])

        key, subkey = random.split(key)

        x0_ind = random.choice(subkey, jnp.arange(P.shape[0], dtype=jnp.int32), 
                               shape=(num_traj,), p=init_dist)

        keys = random.split(key, num=num_traj+1)

        def sample_traj(i, key):
            t_ind = jnp.zeros(max_num_step, dtype=jnp.int32)
            t_ind = t_ind.at[0].set(x0_ind[i])  
            for j in range(1, max_num_step):
                key, subkey = random.split(key)
                next_loc = random.choice(subkey, jnp.arange(P.shape[1], dtype=jnp.int32), shape=(1,), p=P[t_ind[j-1]])
                t_ind = t_ind.at[j].set(next_loc[0])

            return t_ind    

        sample_trajs = jax.vmap(sample_traj, in_axes=(0, 0), out_axes=0)
        sampled_traj_ind = sample_trajs(jnp.arange(len(x0_ind), dtype=jnp.int32), keys[:-1])
        traj_data_ind.append(sampled_traj_ind)

    return traj_data_ind


def plot_time_to_sink(ages, n_points_list, traj_data_ind, sink_idx_list,  ncols=5, imsize=5):
    n_rows = int(len(ages)/ncols) 
    n_rows = n_rows + 1 if len(ages) % ncols != 0 else n_rows 

    fig, axes = plt.subplots(n_rows, ncols, figsize=(ncols * imsize, imsize*n_rows), sharex=True, sharey=True)
    axes = axes.flatten()

    for i in range(len(ages)):
        n_points = n_points_list[i] 
        traj_data_ind_i = traj_data_ind[i]
        sink_idx = sink_idx_list[i] 

        is_in = jnp.isin(traj_data_ind_i, sink_idx)
        ft = jnp.argmax(is_in, axis=1) # first time sink is reached
        no_match = jnp.all(is_in == False, axis=1) # rows with no sinks
        ft = ft.at[no_match].set(traj_data_ind_i.shape[1]+1) # set to trajectory length + 1 if no sink is reached

        axes[i].hist(ft, bins=range(0, traj_data_ind_i.shape[1]+2), alpha=1, color='blue', edgecolor='black')
        axes[i].set_xlabel('Steps to reach sink')
        axes[i].set_ylabel('Number of trajectories')
        axes[i].set_title('Age ' + str(ages[i]))
        print(f'Age {ages[i]} : Steps to reach sink\n' + 'Length of flows: ' + str(traj_data_ind_i.shape[1]) + ' steps' + 
                  f'\nPercentage of particles that did not reach sink: {jnp.sum(no_match) / len(traj_data_ind_i) * 100:.2f} %' + 
                  f'\n Coverage: {len(np.unique(traj_data_ind_i))/n_points * 100 if n_points > 0  else 0:.2f} %')
    plt.show()


def downsample_adata_by_age(adata, n, time_key, PRNG_KEY=0, chosen_times=None):
    """
    Downsample an AnnData object by age.

    Parameters
    ----------
    adata : anndata.AnnData
        The input AnnData object.
    n : int
        The number of cells to sample at each time point.
    time_key : str
        The key in adata.obs that contains the time information.
    PRNG_KEY : int or jax.random.PRNGKey, optional
        The random key or seed for reproducibility. Default is 0.

    Returns
    -------
    anndata.AnnData
        A downsampled AnnData object.
    """
    
    if type(PRNG_KEY) is int:
        key = jax.random.PRNGKey(PRNG_KEY) 
    else:
        key = PRNG_KEY
    
    if chosen_times is None:
        times = sorted(adata.obs[time_key].unique())
    else:
        times = chosen_times

    n_per_time = n
    
    sampled_indices = []
    
    for t in times:
        indices = np.where(adata.obs[time_key] == t)[0]
        
        if len(indices) <= n_per_time:
            sampled_indices.extend(indices)
        else:
            subkey, key = jax.random.split(key)
            sampled = jax.random.choice(subkey, indices, shape=(n_per_time,), replace=False)
            sampled_indices.extend(np.array(sampled))
    
    
    sampled_adata = adata[sampled_indices].copy()
    
    
    
    return sampled_adata, key



@jax.jit
def cost(f,g):
    return jnp.sum(jnp.square(f - g))


@jax.jit
def compute_traj_cost(A0, A1, batch_size=1500):
    cost_mat = jnp.zeros((A0.shape[0], A1.shape[0]))

    map = jax.jit(jax.vmap(jax.vmap(cost, in_axes=(0,None)), in_axes=(None, 0)))
    batch_size = min(batch_size, A0.shape[0], A1.shape[0])
    for i in range(0, A0.shape[0], batch_size):
        for j in range(0, A1.shape[0], batch_size):
            val = map(A1.at[j:j+batch_size].get(), A0.at[i:i+batch_size].get())
            cost_mat = cost_mat.at[i:i+batch_size, j:j+batch_size].set(val)

    return cost_mat / A0.shape[1]


def run_sweep(T, T_dict, n, num_experiments, dt, true_dt, num_step, num_traj, adata, adata_keys, exp_dir, StatOT_params, gStatOT_params, all_res_df,
              max_gStatOT_iter=20_000, max_StatOT_iter=20_000, key=42, HDR_cutoff=0, dtype=jnp.float64, save_adatas=False, constraint_tol=1e-5):

    chosen_times = T_dict[T]
    for exp_num in range(num_experiments):

        down_sampled_adata, key = downsample_adata_by_age(adata, n=n, time_key=adata_keys['time_key'], PRNG_KEY=key, chosen_times=chosen_times)

        if gStatOT_params is not None:
            print("\n Running gStatOT experiments ---------")
            model_params = gStatOT_params[T]
            lam_vals = model_params['lam']
            w_vals = model_params['w']
            eps2_vals = model_params['epsilon2']
            r_vals = model_params['r']
            for lam in lam_vals:
                for w in w_vals:
                    for eps2 in eps2_vals:
                        for r in r_vals:

                            gOT_down_sampled_adata = down_sampled_adata.copy()

                            gSOT = gStatOT.gStatOT(adata=gOT_down_sampled_adata, adata_keys=adata_keys,
                                                   dt=dt, cost_scaling='mean', dtype=dtype)

                            model_params = {'lam': lam,
                                            'w' : w,
                                            'epsilon2': eps2, 
                                            'epsilon1': 0.005,
                                            'epsilon3': 0.005,
                                            'r': r}

                            print("Running gStatOT with params: ", model_params) 
                            method_str = 'gStatOT_' + 'lam=' + str(model_params['lam']) + '_' + 'w=' + str(model_params['w']) + '_' + 'eps2=' + str(model_params['epsilon2']) + '_' + 'r=' + str(model_params['r'])

                            if exp_dir is not None: 
                                save_dir = exp_dir + '/gStatOT/' + f'T_{T}/{method_str}/{exp_num}/'
                            else:
                                save_dir = None

                            gSOT.set_save_dir(save_dir=save_dir)
    
                            gSOT.fit(model_params=model_params, max_iter=max_gStatOT_iter, verbose=False, 
                                     constraint_tol=constraint_tol)

                            gSOT.get_lin_fate_probs(label_key=adata_keys['cell_type_key'], 
                                                    all_labels=np.unique(adata.obs[adata_keys['cell_type_key']]), HDR_cutoff=HDR_cutoff)
                            gSOT.get_trajectories(num_step=num_step, num_traj=num_traj, plot_hitting_time=False, 
                                                  plot_traj=False)
    
                            if save_adatas:
                                gSOT.adata.write_h5ad(save_dir + 'gStatOT_fitted_adata.h5ad') 

                            metric_tests = metrics.Metric_Evaluator(method=method_str,
                                                                    test_adata=gOT_down_sampled_adata,
                                                                    true_adata=adata,
                                                                    time_key=adata_keys['time_key'],
                                                                    embed_key=adata_keys['embed_key'],
                                                                    exp_dir=save_dir,
                                                                    plot_metrics=False)
    
                            metric_tests.w2_marginal_error()
                            metric_tests.fp_tv_error(label_key=adata_keys['cell_type_key'])
                            metric_tests.w2_trajectory_error(test_dt=dt, true_dt=true_dt)
                            gOT_df = metric_tests.results_df
                            gOT_solver_df = gSOT.solver_df
                            gOT_df['n_iter']  = gOT_solver_df['n_iter'].values[0]
                            gOT_df['opt_time'] = gOT_solver_df['optimization time'].values[0]
                            gOT_df['const_err'] = gOT_solver_df['max constraint error'].values[0]

                            gOT_df['experiment'] = exp_num
                            gOT_df['N'] = T * n
                            gOT_df['T'] = T
                            all_res_df = pd.concat([all_res_df, gOT_df], ignore_index=True)
                            if exp_dir is not None:
                                all_res_df.to_csv(exp_dir + 'sweep_results.csv')

                            # free up memory
                            del gOT_down_sampled_adata
                            del gSOT
                            del metric_tests
                            jax.clear_caches()
                            gc.collect()


        if StatOT_params is not None:
            print("\n Running StatOT experiments ---------")
            sot_down_sampled_adata = down_sampled_adata.copy()
            sOT = StatOT.StatOT(adata=sot_down_sampled_adata, adata_keys=adata_keys, dt=dt, dtype=dtype)
            eps_vals = StatOT_params[T]['epsilon']
            for eps in eps_vals:
            
                sOT_params = {'epsilon': eps,
                              'lse': True,
                              'cost_scaling': 'mean'}
                
                method_str = 'StatOT_' + 'eps=' + str(sOT_params['epsilon'])

                if exp_dir is not None: 
                    save_dir = exp_dir + '/StatOT/' + f'T_{T}/{method_str}/{exp_num}/'
                else:
                    save_dir = None
    
                sOT.fit(model_params=sOT_params, max_iter=max_StatOT_iter, verbose=False, constraint_tol=constraint_tol)
                
                sOT.get_lin_fate_probs(label_key=adata_keys['cell_type_key'], all_labels=np.unique(adata.obs[adata_keys['cell_type_key']]))
                sOT.get_trajectories(num_step=num_step, num_traj=num_traj, plot_hitting_time=False, plot_traj=False)
    
                sot_metric_tests = metrics.Metric_Evaluator(method=method_str,
                                                        test_adata=sot_down_sampled_adata,
                                                        true_adata=adata,
                                                        time_key=adata_keys['time_key'],
                                                        embed_key=adata_keys['embed_key'],
                                                        exp_dir=save_dir,
                                                        full_supp=False,
                                                        plot_metrics=False)
                
                sot_metric_tests.w2_marginal_error()
                sot_metric_tests.fp_tv_error(label_key=adata_keys['cell_type_key'])
                sot_metric_tests.w2_trajectory_error(test_dt=dt, true_dt=true_dt)
                sOT_df = sot_metric_tests.results_df
                sOT_solver_df = sOT.solver_df
                sOT_df['const_err'] = sOT_solver_df['inf_norm_error'].values
                sOT_df['experiment'] = exp_num
                sOT_df['N'] = T * n
                sOT_df['T'] = T
                all_res_df = pd.concat([all_res_df, sOT_df], ignore_index=True)

                if exp_dir is not None:
                    all_res_df.to_csv(exp_dir + 'sweep_results.csv')

    return all_res_df