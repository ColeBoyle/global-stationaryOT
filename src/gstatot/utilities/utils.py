import jax
import jax.numpy as jnp
from jax import vmap, jit
from jax import random
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

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
    X_sq_norms = jnp.sum(X * X, axis=1)
    Z_sq_norms = jnp.sum(Z * Z, axis=1)
    squared_distances = Z_sq_norms[:, None] + X_sq_norms[None, :] - 2 * Z @ X.T
    return jnp.maximum(squared_distances, 0)

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

def plot_trajectories(method, adata, N, ncols=5, imsize=2, prefix="", sup_title=None, save_path=None, a=0.8, embed_key='X_pca', traj_data_ind=None):

    ages = list(adata.uns[f'{method}_traj_data'].keys())
    traj_data = [adata.obsm[embed_key][traj_data_ind[i]] for i in range(len(ages))]
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

        P = jnp.asarray(transition_matrices[i], dtype=jnp.float32)

        init_dist = jnp.asarray(init_dists[i], dtype=jnp.float32)

        key, subkey = random.split(key)

        x0_ind = random.choice(subkey, jnp.arange(P.shape[0], dtype=jnp.int32), 
                               shape=(num_traj,), p=init_dist).astype(jnp.int32)

        keys = random.split(key, num=num_traj+1)

#        @jit
#        def sample_traj(i, key):
#            t_ind = jnp.zeros(max_num_step, dtype=jnp.int32)
#            t_ind = t_ind.at[0].set(x0_ind[i])  
#            for j in range(1, max_num_step):
#                key, subkey = random.split(key)
#                next_loc = random.choice(subkey, jnp.arange(P.shape[1], dtype=jnp.int32), shape=(1,), p=P[t_ind[j-1]])
#                t_ind = t_ind.at[j].set(next_loc[0])
#
#            return t_ind    
#
#        sample_trajs = jax.vmap(sample_traj, in_axes=(0, 0), out_axes=0)
#        sampled_traj_ind = sample_trajs(jnp.arange(len(x0_ind), dtype=jnp.int32), keys[:-1])
        @jax.jit
        def sample_traj(x0, key):
            step_keys = random.split(key, max_num_step - 1)

            def step_fn(current_loc, step_key):
                next_loc = random.choice(step_key, P.shape[1], p=P[current_loc]).astype(jnp.int32)
                return next_loc, next_loc

            _, traj_tail = jax.lax.scan(step_fn, init=x0, xs=step_keys)

            full_traj = jnp.concatenate([jnp.array([x0], dtype=jnp.int32), traj_tail])

            return full_traj

        sample_trajs = jax.vmap(sample_traj, in_axes=(0, 0))
        sampled_traj_ind = sample_trajs(x0_ind, keys[:-1])

        traj_data_ind.append(sampled_traj_ind)

    return traj_data_ind


def plot_time_to_sink(ages, n_points_list, traj_data_ind, sink_idx_list,  ncols=5, imsize=5, return_traj_len=False):
    n_rows = int(len(ages)/ncols) 
    n_rows = n_rows + 1 if len(ages) % ncols != 0 else n_rows 

    fig, axes = plt.subplots(n_rows, ncols, figsize=(ncols * imsize, imsize*n_rows), sharex=True, sharey=True)
    axes = axes.flatten()
    traj_lens = []
    for i in range(len(ages)):
        n_points = n_points_list[i] 
        traj_data_ind_i = traj_data_ind[i]
        sink_idx = sink_idx_list[i] 

        is_in = jnp.isin(traj_data_ind_i, sink_idx)
        ft = jnp.argmax(is_in, axis=1) # first time sink is reached
        no_match = jnp.all(is_in == False, axis=1) # rows with no sinks
        ft = ft.at[no_match].set(traj_data_ind_i.shape[1]+1) # set to trajectory length + 1 if no sink is reached
        traj_lens.append(ft)

        axes[i].hist(ft, bins=range(0, traj_data_ind_i.shape[1]+2), alpha=1, color='blue', edgecolor='black')
        axes[i].set_xlabel('Steps to reach sink')
        axes[i].set_ylabel('Number of trajectories')
        axes[i].set_title('Age ' + str(ages[i]))
        print(f'Age {ages[i]} : Steps to reach sink\n' + 'Length of flows: ' + str(traj_data_ind_i.shape[1]) + ' steps' + 
                  f'\nPercentage of particles that did not reach sink: {jnp.sum(no_match) / len(traj_data_ind_i) * 100:.2f} %' + 
                  f'\n Coverage: {len(np.unique(traj_data_ind_i))/n_points * 100 if n_points > 0  else 0:.2f} %')
    plt.show()
    
    if return_traj_len:
        return np.array(traj_lens)


def downsample_adata_by_age(adata, n, time_key, PRNG_KEY=0, chosen_times=None,
                            batch_ind=None):
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
    batch_ind : int, optional
        Zero-based index of a disjoint batch. When provided, each time point is
        permuted reproducibly and the corresponding block of ``n`` cells is
        selected.

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
        times = np.unique(adata.obs[time_key]) 
    else:
        times = chosen_times

    n_per_time = n

    if batch_ind is not None and batch_ind < 0:
        raise ValueError("batch_ind must be non-negative")

    batch_start = batch_ind * n_per_time if batch_ind is not None else None
    batch_stop = batch_start + n_per_time if batch_start is not None else None
    
    sampled_indices = []
    
    for t in times:
        indices = np.where(adata.obs[time_key] == t)[0]

        if batch_ind is not None:
            if batch_stop > len(indices):
                raise ValueError(
                    f"Batch {batch_ind} requires at least {batch_stop} cells "
                    f"at time {t}, but only {len(indices)} are available"
                )
            subkey, key = jax.random.split(key)
            permuted = jax.random.permutation(subkey, indices)
            sampled_indices.extend(np.array(permuted[batch_start:batch_stop]))
        elif len(indices) <= n_per_time:
            sampled_indices.extend(indices)
        else:
            subkey, key = jax.random.split(key)
            sampled = jax.random.choice(subkey, indices, shape=(n_per_time,), replace=False)
            sampled_indices.extend(np.array(sampled))
    
    
    sampled_adata = adata[sampled_indices].copy()
    
    
    
    return sampled_adata

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



def dropout(adata, dropout_cutoff, dropout_prob, time_key='age', key=0):
    """
    Randomly drop out values in adata.X that are below a certain cutoff quantile with a given probability.

    Parameters
    ----------
    adata : anndata.AnnData
        The input AnnData object.
    dropout_cutoff : float
        The cutoff value below which entries will be considered for dropout.
    dropout_prob : float
        The probability with which to drop out entries below the cutoff.

    Returns
    -------
    anndata.AnnData
        An AnnData object with dropout applied to adata.X.
    """
    
    if type(key) is int:
        key = jax.random.PRNGKey(key)  

    X_drop = adata.X.copy()
    times = np.unique(adata.obs[time_key])
    # drop at each age idepndently
    for t in times:
        t_indices = np.where(adata.obs[time_key] == t)[0]
        X_t = adata.X[t_indices]
        quantiles = np.quantile(X_t, dropout_cutoff, axis=0) # shape (n_genes,)
        mask = X_t < quantiles # 
        key, subkey = jax.random.split(key)
        random_vals = jax.random.uniform(subkey, shape=X_t.shape)

        dropout_mask = (random_vals < dropout_prob) & mask

        X_drop[t_indices] = np.where(dropout_mask, 0, X_t) # set dropped values to 0, keep others the same

    adata.X = X_drop
    
    return adata


def get_lin_fate_probs(model, label_key, full_supp, all_labels=None, 
                       lin_fp_error_tol=1e-2, init_HDT_cutoff=0.00, num_restarts=10):

    if all_labels is None:
        all_labels = np.unique(model.adata.obs[label_key])
    model.adata.uns[f'{label_key}_fp_labels'] = all_labels

    if not full_supp:
        model.adata.obsm[f'{label_key}_fp'] = np.nan * np.ones((model.adata.n_obs, len(all_labels)), dtype=np.float32)

    jax.config.update("jax_enable_x64", True)
    for i in range(model.T):
        growth_i = model.all_growth[i]
        if full_supp:
            pi = model.adata.obsp[f'pi_{model.times[i]}']
            labels = model.adata.obs[label_key].to_numpy()

        else:
            pi = model.adata.uns[f'pi_{model.times[i]}']
            labels = model.adata[model.adata.obs[model.time_key] == model.times[i]].obs[label_key].to_numpy()

        sink_idx = growth_i < 1

        if sink_idx.sum() == 0:
            print(f"No sinks at time {model.times[i]}, cannot compute fate probabilities.")
            lin_fp = np.nan * np.ones((labels.shape[0], all_labels.shape[0]))
            if full_supp:
                model.adata.obsm[f'{label_key}_fp_t={model.times[i]}'] = lin_fp
            else:
                model.adata.obsm[f'{label_key}_fp'][model.adata.obs[model.time_key] == model.times[i]] = lin_fp
            continue

        restarts = num_restarts
        computed = False
        HDT_cutoff = init_HDT_cutoff
        HDT_increment = 1e-2
#         HDR_cutoff = 1e-3
        while (restarts > 0):

            if HDT_cutoff > 0:
                # remove the states with the lowest probability transitions 
                # Trim distribution to HDR: remove lowest probability states with collective mass < HDT_cutoff
                # i.e. for more stable computation of fate probabilities we restrict to states containing 99.9% of 
                # the probability mass

                trans_mass = pi - jnp.diag(jnp.diag(pi)) 
                norm_trans_mass = trans_mass / jnp.sum(trans_mass)
                mass_in = norm_trans_mass.sum(0)
                mass_out = norm_trans_mass.sum(1)
                dist = mass_in + mass_out
                dist = jnp.where(sink_idx, dist * 2, dist) 
                
                dist_ordered, dist_idx = jnp.sort(dist), jnp.argsort(dist)
                cumsum_dist = jnp.cumsum(dist_ordered)
                cutoff_idx = jnp.searchsorted(cumsum_dist, HDT_cutoff)
                low_mass_idx = dist_idx[:cutoff_idx]
                non_zero_idx = jnp.ones(pi.shape[0], dtype=bool).at[low_mass_idx].set(False)

#             if HDR_cutoff > 0:
#                 # Trim distribution to HDR: remove lowest probability states with collective mass < HDR_cutoff
#                 # i.e. for more stable computation of fate probabilities we restrict to states containing 99.9% of 
#                 # the probability mass
#                 dist = pi.sum(0)
#                 dist_ordered, dist_idx = jnp.sort(dist), jnp.argsort(dist)
#                 cumsum_dist = jnp.cumsum(dist_ordered)
#                 cutoff_idx = jnp.searchsorted(cumsum_dist, HDR_cutoff)
#                 low_mass_idx = dist_idx[:cutoff_idx]
#                 non_zero_idx = jnp.ones(pi.shape[0], dtype=bool).at[low_mass_idx].set(False)
#     #            non_zero_idx = np.asarray(mask)
            else:
                zero_states = pi.sum(0) == 0
                zero_rows = pi.sum(1) == 0 # drop out from underflow
                non_zero_idx = np.logical_and(~zero_states, ~zero_rows)

            orig_inds = np.arange(pi.shape[0])
            pi_non_zero = pi[non_zero_idx][:, non_zero_idx]
            new_inds = orig_inds[non_zero_idx]

            # remove any new zero rows/columns that may have been created by the first round of filtering
            while True:
                zero_cols = pi_non_zero.sum(axis=0) == 0
                zero_rows = pi_non_zero.sum(axis=1) == 0

                if not zero_rows.any() and not zero_cols.any():
                    break

                valid_idx = ~(zero_rows | zero_cols)

                pi_non_zero = pi_non_zero[valid_idx][:, valid_idx]            
                new_inds = new_inds[valid_idx]
            labels_non_zero = labels[new_inds]
            sink_idx_non_zero = sink_idx[new_inds]  
            sink_gr_i = - model.all_growth_rates[i][new_inds][sink_idx_non_zero] * model.dt

            # set sink matrix
            label_to_col = {label: j for j, label in enumerate(all_labels)}
            S = np.zeros((pi_non_zero.shape[0], all_labels.shape[0]), dtype=np.float64)
            for sink_row, gr, lbl in zip(np.where(sink_idx_non_zero)[0], sink_gr_i, labels_non_zero[sink_idx_non_zero]):
                if lbl in label_to_col:
                    S[sink_row, label_to_col[lbl]] = gr
            assert np.all(S>=0)

            P = row_normalize(pi_non_zero)
            P = np.hstack((P, S))
            ZI = np.hstack((np.zeros((S.shape[1], pi_non_zero.shape[1])), np.eye(S.shape[1])))
            P = np.vstack((P, ZI))
            P = row_normalize(jnp.asarray(P, jnp.float64))

            sink_idx_aug = np.zeros(P.shape[0], dtype=bool)
            sink_idx_aug[pi_non_zero.shape[0]:] = True
            labels_aug = np.concatenate((labels_non_zero, all_labels))

            redo = False
            try:
                lin_fp, sink_labels = compute_fate_probs_lineages(P, sink_idx_aug, labels_aug)
                
            except np.linalg.LinAlgError as e:
                print(f"Age {model.times[i]}: LinAlgError during fate probability computation: {e}. {restarts} restarts remaining.")
                redo = True
               
            if not redo:


                if len(lin_fp.shape) < 2:
                    lin_fp = lin_fp.reshape(-1, 1)


                lin_fp_error = np.max(np.abs(lin_fp.sum(1) - 1.0))

                if len(lin_fp) == 0:
                    print(f"Age {model.times[i]}: Restarting due to empty lineage matrix. {restarts} restarts remaining.")
                    redo = True

                elif np.any(np.isnan(lin_fp)):
                    print(f"Age {model.times[i]}: Restarting due to NaNs in lineage matrix. {restarts} restarts remaining.")
                    redo = True

                elif (lin_fp_error > lin_fp_error_tol):
                    print(f"Age {model.times[i]}: Restarting due to high fate prob error: {lin_fp_error:.5e} (tol={lin_fp_error_tol}). {restarts} restarts remaining.")
                    redo = True

            if redo:
                restarts -= 1
                HDT_cutoff += HDT_increment

            else:
                computed = True
                break

        if not computed:
            lin_fp = np.nan * np.ones((labels.shape[0], all_labels.shape[0]))
            print(f"Failed to compute fate probabilities at time {model.times[i]} after {restarts} restarts, setting to NaN.")

            if full_supp:
                model.adata.obsm[f'{label_key}_fp_t={model.times[i]}'] = lin_fp
            else:
                model.adata.obsm[f'{label_key}_fp'][model.adata.obs[model.time_key] == model.times[i]] = lin_fp

            continue
        
        # remove the augmented sink states for the output
        lin_fp = lin_fp[:pi_non_zero.shape[0], :]
    

        lin_fp = np.asarray(row_normalize(lin_fp), dtype=np.float32)
        # add a 0 column for cell types with no sinks
        fp_lin_all= np.zeros((lin_fp.shape[0], len(all_labels)), dtype=np.float32)
        for j, ct in enumerate(all_labels):
            if ct in sink_labels:
                idx = np.where(sink_labels == ct)[0][0]
                fp_lin_all[:, j] = lin_fp[:, idx]
        # add back in  0's for uncomputed points
        fp_lin_all_full = np.zeros((labels.shape[0], len(all_labels)), dtype=np.float32)
        fp_lin_all_full[new_inds] = fp_lin_all[: len(labels_non_zero)]

        if full_supp:
            model.adata.obsm[f'{label_key}_fp_t={model.times[i]}'] = fp_lin_all_full
        else:
            model.adata.obsm[f'{label_key}_fp'][model.adata.obs[model.time_key] == model.times[i]] = fp_lin_all_full


    if model.dtype != jnp.float64:
        jax.config.update("jax_enable_x64", False)


@jax.jit
def compute_mfpt_to_set_jax(P, sink_mask):
    """
    Compute standard MFPT to a single set of sink states.
    
    :param P: (N, N) transition matrix.
    :param sink_mask: (N,) boolean array, True for the target/sink states.
    :return: (N,) vector containing the MFPTs.
    """
    I = jnp.eye(P.shape[0])
    
    Pp = jnp.where(sink_mask[:, None], I, I - P)
    
    # (I-P)tau' = 1 
    taup = jnp.where(sink_mask, 0.0, 1.0)
    
    t = jnp.linalg.solve(Pp, taup)
    return t

def get_mfpt(model, full_supp, dt=1.0):

    if full_supp:
        model.adata.obsm[f'mfpt'] = np.nan * np.ones((model.adata.n_obs, model.T), dtype=np.float32)
    else:
        model.adata.obs[f'mfpt'] = np.nan * np.ones(model.adata.n_obs, dtype=np.float32)

    for i in range(model.T):
        if full_supp:
            pi = model.adata.obsp[f'pi_{model.times[i]}']

            sink_idx = model.all_growth[i] < 1
            HDT_cutoff = 0.05
            trans_mass = pi - jnp.diag(jnp.diag(pi)) 
            norm_trans_mass = trans_mass / jnp.sum(trans_mass)
            mass_in = norm_trans_mass.sum(0)
            mass_out = norm_trans_mass.sum(1)
            dist = mass_in + mass_out
            # double the sink mass since it's only counted in the row sum
            dist = jnp.where(sink_idx, dist * 2, dist) 
            
            dist_ordered, dist_idx = jnp.sort(dist), jnp.argsort(dist)
            cumsum_dist = jnp.cumsum(dist_ordered)
            cutoff_idx = jnp.searchsorted(cumsum_dist, HDT_cutoff)
            low_mass_idx = dist_idx[:cutoff_idx]
            non_zero_idx = jnp.ones(pi.shape[0], dtype=bool).at[low_mass_idx].set(False)
            sink_idx = sink_idx[non_zero_idx] 
            pi = pi[non_zero_idx][:, non_zero_idx]
            
        else:
            pi = model.adata.uns[f'pi_{model.times[i]}']
            sink_idx = model.all_growth[i] < 1

        P = row_normalize(pi)
        mfpt = compute_mfpt_to_set_jax(P, sink_idx)
        if full_supp:
            model.adata.obsm[f'mfpt'][:, i][non_zero_idx] = mfpt * dt
        else:
            age_mask = model.adata.obs[model.time_key] == model.times[i]
            model.adata.obs.loc[age_mask, f'mfpt'] = mfpt * dt

from matplotlib.colors import to_rgb

def plot_fp(adata, age, time_key, embed_key, label_key, lineages, lineage_color_map, full_supp, ax=None, max_s=20, x_lim=None, y_lim=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if full_supp:
        probabilities = adata.obsm[f'{label_key}_fp_t={age}']
        dist = adata.obsp[f'pi_{age}'].sum(0)
        s = (dist) / (dist.max() ) * max_s
    else:
        probabilities = adata.obsm[f'{label_key}_fp'][adata.obs[time_key] == age]
        s = max_s

    zero_row_mask = probabilities.sum(axis=1) == 0
    probabilities = probabilities[~zero_row_mask]

    rgb_values = np.array([to_rgb(lineage_color_map[l]) for l in lineages])
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    max_prob = np.nanmax(probabilities, axis=1)
    base_colors = probabilities @ rgb_values

    grey_target = np.array([0.8, 0.8, 0.8])

    n_lineages = probabilities.shape[1]
    min_possible_prob = 1.0 / n_lineages
    certainty = np.clip((max_prob - min_possible_prob) / (1.0 - min_possible_prob), 0, 1)
    certainty = certainty[:, np.newaxis]
    blended_colors = (certainty * base_colors) + ((1.0 - certainty) * grey_target)

    if full_supp:
        X = adata.obsm[embed_key][~zero_row_mask]
    else:
        X = adata[adata.obs[time_key] == age].obsm[embed_key][~zero_row_mask]

    ax.scatter(X[:, 0], X[:, 1], c=blended_colors, s=s, edgecolor='none')
    if x_lim is not None and y_lim is not None:
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
#    ax.axis('off')
    if ax is None:
        plt.show()