import jax
import sde_simulator as sde_sim
import jax.numpy as jnp
import numpy as np
import os

init_seed = 0
dim = 10
save_dir = '../../data/sim_data/bistable_simulations'
os.makedirs(save_dir, exist_ok=True)
num_progenitors = 5
sink_rad = 0.3

time_step = 1e-2 # days
c = 0.005 #  aging rate
n_iter = 10e3 # 100 days or 6 months in biological age (c * n_iter * time_step = 0.5 years)

sde_sim_parameters = {
                "sigma^2": 0.1,
                "time_step": 1e-2,
                "dim": dim,
                "max_num_iter": n_iter,
                "max_pop":  2e3,
                "sample_rate": 100, # every 100 iterations, or 1 day
            }

fp_n_traj_parameters = {
    'max_num_traj_itr': 1000, # (10 days) maximum number of iterations to run sim for trajectory sampling; cells generally last 2 days
    'max_pop_buffer': 2000, # add buffer to max_pop to ensure we allow for all possible cell growth during the fp calculation 
    'num_traj_samples': 500,
}

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


def death(X, t):
    dr = 5
    return jnp.where((jnp.linalg.norm(X-z0(t)) < sink_rad) | (jnp.linalg.norm(X-z1(t)) < sink_rad), dr, 0.0)

def birth(X, t):
    gr = 100 
    a = jnp.zeros((dim,))
    A = jnp.where((jnp.linalg.norm(X-a) == 0), gr, 0.0) # source gr
    B = jnp.where((jnp.linalg.norm(X-z0(t)) > 0.3) & (X[1] > 0),  c * t * X[1], 0.0) # +y gr
    return A + B

def potential(X, t):
    return - 1/2 * jnp.linalg.norm(X - z0(t))**2 * jnp.linalg.norm(X - z1(t))**2 




S = sde_sim.Simulation(potential=potential, drift=None, birth=birth, death=death, 
                       sde_sim_parameters=sde_sim_parameters)

X0 = jnp.zeros((num_progenitors,dim))

for seed in [init_seed + i for i in range(11)]:

    adata = S.make_sim_adata(seed, X0=X0, sample_traj=True, get_fp=True, **fp_n_traj_parameters)
    adata.obs['chronological_age'] = adata.obs['time'] # rename time to chronological age for clarity
    del adata.obs['time']
    adata.obs['cell_type'] = 'Progenitor'  
    X = adata.X

    for time in jnp.unique(adata.obs['chronological_age'].to_numpy()):
        mask = adata.obs['chronological_age'] == time
        sink_y_R = z0(time)
        sink_y_L = z1(time)
        cond_1 =  (jnp.linalg.norm(X - sink_y_R, axis=1) < sink_rad) 
        cond_2 =  (jnp.linalg.norm(X - sink_y_L, axis=1) < sink_rad)
        adata.obs.loc[mask & cond_1, 'cell_type'] = '+y'
        adata.obs.loc[mask & cond_2, 'cell_type'] = '-y'
    
        adata.obs.loc[mask, 'sink'] = (adata.obs.loc[mask, 'growth_rate'] < 0).astype(str)
        adata.obs.loc[mask, 'source'] = (adata.obs.loc[mask, 'growth_rate'] > 0).astype(str)

    @jax.jit
    def type_cond_fn(coords, time):
        sink_y_R = z0(time)
        sink_y_L = z1(time)
        cond_1 =  (jnp.linalg.norm(coords - sink_y_R, axis=1) < sink_rad) 
        cond_2 =  (jnp.linalg.norm(coords - sink_y_L, axis=1) < sink_rad)
        return jnp.where(cond_1, 0, # +y 
                         jnp.where(cond_2, 1,  # -y
                                   2))  # Progenitor

    S.compute_fate_probabilities(key=jax.random.PRNGKey(seed+100), adata=adata,
                                 type_cond_fn=type_cond_fn,
                                 type_label='cell_type', time_col='chronological_age', 
                                 max_num_itr=fp_n_traj_parameters['max_num_traj_itr'], 
                                 max_pop_buffer=fp_n_traj_parameters['max_pop_buffer'])

    adata.uns['all_cell_types'] = np.unique(adata.obs['cell_type'])
    adata.obsm['X_pca'] = adata.X

    adata.write_h5ad(os.path.join(save_dir, f'bistable_sim_seed={seed}.h5ad'))