## Ran with python-3.13.2, gcc-13.3
import os
import sys
import jax #==0.6.0
import jax.numpy as jnp
import numpy as np #=2.1.1

import sde_simulator as sde_sim
import anndata #==0.11.3
from gstatot import StatOT


jax.config.update('jax_platform_name', 'cpu')

if len(sys.argv) > 1:
    sim_num = int(sys.argv[1])
else:
    sim_num = 0

seed = sim_num
sim_name = f'bistable_sim_{sim_num}'
dim = 10
num_progenitors = 5
sink_rad = 0.3
c = 0.005
n_init_traj_samples = 500

n_iter = 10e3
get_traj = True
get_fate_probs = True
run_sim = True

data_folder = f"../../data/sim_data/"
os.makedirs(data_folder, exist_ok=True)

sde_sim_parameters = {
                "sigma^2": 0.1,
                "time_step": 1e-2,
                "dim": dim,
                "max_num_iter": n_iter,
                "max_pop":  25e3,
                "sample_rate": 100,
                "print_rate": n_iter,
                "seed": seed
            }

# helper for formatting trajectories
def forward_fill_points_shift(arr):

    out = arr.copy()
    n_rows, row_len, point_dim = out.shape

    mask = ~np.isnan(out).all(axis=2)

    for i in range(n_rows):
        valid_points = out[i][mask[i]]
        num_valid = len(valid_points)
        if num_valid < row_len:
            out[i] = np.vstack([
                valid_points,
                np.full((row_len - num_valid, point_dim), np.nan)
            ])
        else:
            out[i] = valid_points

    mask = ~np.isnan(out).all(axis=2)
    idx = np.where(mask, np.arange(row_len)[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = out[np.arange(n_rows)[:, None], idx]

    return out

def get_sim_functions():
    import jax
    import jax.numpy as jnp
    from jax import jit
    def z0(t):
        x0 = 1
        y0 = 1.25
        z0 = 0
        return jnp.concat([jnp.array([x0,  y0, z0]) + c * t * jnp.array([1, 0, 1]), jnp.zeros((dim-3,))])
    
    def z1(t):
        x0 = 1
        y0 = -1.25
        z0 = 0
        return jnp.concat([jnp.array([x0, y0, z0]) + c * t * jnp.array([1, 0, -1]), jnp.zeros((dim-3,))])
    
    
    def _death(X, t):
        dr = 5
        return jnp.where((jnp.linalg.norm(X-z0(t)) < sink_rad) | (jnp.linalg.norm(X-z1(t)) < sink_rad), dr, 0.0)
    
    def _birth(X, t):
        gr = 100 + c * t
        a = jnp.zeros((dim,))
        A = jnp.where((jnp.linalg.norm(X-a) == 0), gr, 0.0)
        B = jnp.where((jnp.linalg.norm(X-z0(t)) > 0.3) & (X[1] > 0),  c * t * X[1], 0.0)
        return A + B
    
    birth = jax.vmap(_birth, in_axes=(0, None))
    death = jax.vmap(_death, in_axes=(0, None))
    
    @jit
    def growth_rate(X, t):
        return birth(X,t) - death(X,t)
    
    @jit
    def potential(X, t):
        return - 1/2 * jnp.linalg.norm(X - z0(t))**2 * jnp.linalg.norm(X - z1(t))**2 

    return potential, death, birth, growth_rate, z0, z1




if run_sim:

    potential, death, birth, growth_rate, z0, z1 = get_sim_functions()
    S = sde_sim.Simulation(potential=potential, drift=None, birth=birth, death=death, 
                           sde_sim_parameters=sde_sim_parameters, save_sim=True)

    X0 = jnp.zeros((num_progenitors,dim))
    S.sim_sde(X0=X0, t0=0, verbose=True, fixed_birth=True, exact=False, fixed_point=None)

    # format data into anndata

    X = np.concatenate(S.sim_time_series, axis=0)
    sim_adata = anndata.AnnData(X=X)
    time_masks = []
    len_prev = 0
    sim_adata.obs['growth_rate'] = 0.0
    sim_adata.obs['age'] = np.inf
    sim_adata.obs['cell_type'] = 'Progenitor' 
    sim_adata.obs['sink'] = False
    sim_adata.obs['source'] = False

    for i in range(len(S.sim_time)):
    
        mask = np.zeros(X.shape[0], dtype=bool)
        mask[len_prev: len_prev + len(S.sim_time_series[i])] = True
        len_prev = len(S.sim_time_series[i]) + len_prev
        time_masks.append(mask)
        time = np.float32(S.sim_time[i])

        sim_adata.obs.loc[mask, 'growth_rate'] = np.float32(growth_rate(S.sim_time_series[i], time))
        sim_adata.obs.loc[mask, 'age'] = time

        sink_y_R = z0(time)
        sink_y_L = z1(time)
        rad = sink_rad
        cond_1 =  (np.linalg.norm( X - sink_y_R, axis=1) < rad) 
        cond_2 =  (np.linalg.norm( X - sink_y_L, axis=1) < rad)
        sim_adata.obs.loc[mask & cond_1, 'cell_type'] = '+y'
        sim_adata.obs.loc[mask & cond_2, 'cell_type'] = '-y'

        sim_adata.obs.loc[mask, 'sink'] = sim_adata.obs.loc[mask, 'growth_rate'] < 0
        sim_adata.obs.loc[mask, 'source'] = sim_adata.obs.loc[mask, 'growth_rate'] > 0


        sim_adata.obsm['X_pca'] = sim_adata.X
        sim_adata.uns['true_dt'] = S.sde_sim_parameters['time_step']
        sim_adata.uns['sigma^2'] = S.sde_sim_parameters['sigma^2']


    sim_adata.write_h5ad(f"{data_folder}{sim_name}.h5ad")

else:
    sim_adata = anndata.read_h5ad(f"{data_folder}{sim_name}.h5ad")

def traj_sim(point, time, rng):

    sde_sim_parameters['seed'] = rng.integers(0, 2**32 - 1)
    sde_sim_parameters['max_pop'] = len(point)
    sde_sim_parameters['max_num_iter'] = 1_000
    potential, death, birth, growth_rate, z0, z1 = get_sim_functions()
    S = sde_sim.Simulation(potential=potential, drift=None, death=death, birth=None, 
                       sde_sim_parameters=sde_sim_parameters, save_sim=False)
    S.sim_sde(X0=point, t0=time, exact=True, verbose=True)

    return S.sim_time_series, S.sim_time, S


# get trajectories
if get_traj:
    times = np.array(sim_adata.obs['age'].unique())
    rng = np.random.default_rng(seed=seed) 

    print('Sampling trajectories')
    
    sim_adata.uns['traj_data'] = {}
    
    for time in times:
        print(f'Simulating trajs at age {time}')
        growth_ = sim_adata.obs.loc[sim_adata.obs['age'] == time, 'growth_rate'].values
        growth_ = growth_[growth_ > 0]
        growth_dist = growth_ / growth_.sum()
        X0_ind = rng.choice(sim_adata.obs.loc[(sim_adata.obs['age'] == time) & (sim_adata.obs['growth_rate'] > 0)].index, 
                            size=n_init_traj_samples, p=growth_dist, replace=True)
        X0 = sim_adata[X0_ind].X
        traj, sim_time, sim_class = traj_sim(X0, time, rng)
    
        traj = traj[:, ~np.isnan(traj.sum(axis=-1)).all(axis=0)] # remove nan traj
    
        traj = traj.transpose(1, 0, 2)
        traj = np.asarray(traj, dtype=np.float32)
        traj = forward_fill_points_shift(traj) # format traj and fill nan tails with final valid point
    
        sim_adata.uns['traj_data'][str(time)] = traj
    
    sim_adata.write_h5ad(f"{data_folder}{sim_name}.h5ad")


if get_fate_probs:

    adata_keys  = {'time_key': 'age',
               'cell_type_key': 'cell_type',
               'growth_rate_key': 'growth_rate',
               'embed_key': 'X_pca'}

    sOT = StatOT(adata=sim_adata, adata_keys=adata_keys, 
                        dt=sde_sim_parameters['time_step'], dtype=jnp.float32)
    
    sOT_params = {'epsilon': 0.01,
                  'lse': True,
                  'cost_scaling': 'mean'}
    
    sOT.fit(model_params=sOT_params, max_iter=15_000, verbose=True)
    
    sOT.get_lin_fate_probs(label_key=adata_keys['cell_type_key'], 
                           all_labels=np.unique(sim_adata.obs[adata_keys['cell_type_key']]))
    sOT.adata.write_h5ad(f"{data_folder}{sim_name}.h5ad")
