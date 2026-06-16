import scanpy as sc
import numpy as np
import pandas as pd
import jax.numpy as jnp
import io
from jax import jit
import jax
import subprocess 
import sys
from tqdm import tqdm

sys.path.append('../')
import sde_simulator as sde_sim
from gstatot import utils

network_dir_loc = './'
boolODE_loc = '../../../../../BoolODE'
save_dir = '../../../data/sim_data/BoolODE_GRN_simulations'

network = """Gene	Rule
G\t( tf1 ) and not ( tf2 or tf3 or tf4 )
tf1	not ( tf2 )
tf2	( tf1 or tf2 ) and not ( tf3 )
tf3	( tf3 or tf2 ) and not ( tf4 )
tf4	( tf4 or tf3 ) and not ( tfB1 or tfB2 )
tfB1	( tfB1 or tf4 ) and not ( tfB2 )
tfB2	( tfB2 or tf4 ) and not ( tfB1 )
tfB11	( tfB11 or tfB1 )
tfB22	( tfB22 or tfB2 )
D\t( tfB11 or tfB22 or D)
"""

network_tsv = pd.read_csv(io.StringIO(network), sep='\t')
network_tsv.to_csv(f'{network_dir_loc}/network.tsv', sep='\t', index=False)

ics = """Genes	Values
['tf1', 'G', ]	[1, 1]"""

ics_df = pd.read_csv(io.StringIO(ics), sep='\t')
ics_df.to_csv(f'{network_dir_loc}/ics.tsv', sep='\t', index=False)


default_strengths = """Gene1	Gene2	Strength
tf2	tf2	0.8
tf3	tf3	0.8
tf4	tf4	0.8
tfB1	tf4	1.0
tfB2	tf4	0.9
tfB1	tfB2	0.5
tfB2	tfB1	0.5"""

default_strengths_df = pd.read_csv(io.StringIO(default_strengths), sep='\t')
default_strengths_df.to_csv(f'{network_dir_loc}/default_strengths.tsv', sep='\t', index=False)

# run BoolODE to generate model and y0

subprocess.run(["python", f"{boolODE_loc}/boolode.py", "--config", f"{network_dir_loc}/config.yaml"], check=True)

# Change numpy to jax.numpy in model.py so we run the sim with jax and can jit the drift function 
file_path = "./sims/default_simulation/model.py"
with open(file_path, "r") as file:
    content = file.read()

modified_content = content.replace("import numpy as np", "import jax.numpy as np", 1)

with open(file_path, "w") as file:
    file.write(modified_content)

sys.path.append('./sims/default_simulation')
from model import Model 

max_time = 150 # in days

params = pd.read_csv('./sims/default_simulation/parameters.txt', sep='\t', index_col=0, header=0)
params = list(params.to_dict().values())[0]
y0 = np.load('./sims/default_simulation/simulations/y0_0.npy') # load y0 from BoolODE

bifur_str_init = 0.9 
bifur_str_final = 0.5

# align params as in BoolODE
parNames = np.array(sorted(list(params.keys())))
pars = {}
for k, v in params.items():
    pars[k] = v
pars = [pars[k] for k in parNames]

# fit line to go from bifur_str_init to bifur_str_final over the course of the sim
A = np.array([[1, 1], [max_time, 1]])
b_vec = np.array([bifur_str_init, bifur_str_final])
sol = np.linalg.solve(A, b_vec)

m = sol[0]
b = sol[1]

# get the indices of the parameters we want to change
idx_tfB1_tfB2 = jnp.where(parNames == 'k_tfB1_tfB2')[0][0]
idx_tfB2_tfB1 = jnp.where(parNames == 'k_tfB2_tfB1')[0][0]
pars = jnp.asarray(pars)

hill_thresh = 10.0
def reg_str(t):
    return m * t + b

@jit
def param_func(t, pars=pars):
    pars = pars.at[idx_tfB1_tfB2].set(hill_thresh / reg_str(t))
    pars = pars.at[idx_tfB2_tfB1].set(hill_thresh / reg_str(t))
    return pars

drift = jax.jit(lambda x, t: Model(x, t, param_func(t)))

gen_flux = 2 # increase to increase the number of cells in the sim
@jit
def birth(x, t):
    br = gen_flux 
    return  jnp.where(x[1] >= 10.0, br, 0.0) 
    
@jit
def death(x, t):
    dr1 =  0.5 * gen_flux
    A = jnp.where(jnp.all(x < 0.2), 100, 0.0) # kill cells have no protiens or rna signal
    B = jnp.where((x[-1] > 15.0), dr1, 0.0) 
    return A + B 

@jit
def diff_coef(x, t):
    return 1 * jnp.sqrt(jnp.abs(x)) # 10 -> 1, corrects for noise scale in BoolODE, which used dt over sqrt(dt) as std of noise

y0 = jnp.asarray(y0)

init_seed = 0
time_step = 1e-2 # days
sr_days = 5

n_iter = int(max_time / time_step)
sr = int(sr_days / time_step)

sde_sim_parameters = {
                "time_step": time_step,
                "dim": y0.shape[0],
                "max_num_iter": n_iter,
                "max_pop":  10e3,
                "sample_rate": sr
            }

traj_sampling_kwargs = {
    "max_num_traj_itr": 2000,
    "max_pop_buffer": 2000,
    "num_traj_samples": 500,
}

n_progenitors = 5

X0 = jnp.tile(y0, (n_progenitors, 1))

S = sde_sim.Simulation(drift=drift, birth=birth, death=death, diff_coef=diff_coef,
                       sde_sim_parameters=sde_sim_parameters)
    
for seed in tqdm([init_seed + i for i in range(11)]):

    adata = S.make_sim_adata(seed=seed, X0=X0, t0=0,
                             fixed_birth=True, 
                             sample_traj=False, 
                             get_fp=False, 
                             ensure_pos=True, # same positivity constraint as BoolODE
                             **traj_sampling_kwargs) 
    
    adata.obs['cell_type'] = 'Progenitor'
    gene_names = network_tsv['Gene'].values
    # insert protein names as var names in adata it goes gene protein gene protein ...
    for i in range(adata.shape[1]):
        if i % 2 == 0:
            adata.var_names.values[i] = gene_names[i // 2]
        else:
            adata.var_names.values[i] = gene_names[i // 2] + '_protein'
    

    all_names = np.array(adata.var_names)
    # need it to be numeric for jit, so convert to indices and then back to names after jit
    all_names_indices = jnp.arange(len(all_names))
    tfB11_idx = jnp.where(all_names == 'tfB11')[0][0]
    tfB22_idx = jnp.where(all_names == 'tfB22')[0][0]

    @jit
    def type_cond_fn(coords, time):
        tfB11 = coords[:, tfB11_idx].flatten()
        tfB22 = coords[:, tfB22_idx].flatten()
        return jnp.where((tfB11 > 1.0) & (tfB22 <= 1.0), 0, # B1
                        jnp.where((tfB22 > 1.0) & (tfB11 <= 1.0), 2, # B2
                                 jnp.where((tfB11 > 1.0) & (tfB22 > 1.0), 1, -1))) # B1+B2 or unassigned

    B1_cond = adata[:, 'tfB11'].X.flatten() > 1.0 # express B1 marker
    B2_cond = adata[:, 'tfB22'].X.flatten() > 1.0 # express B2 marker

    adata.obs.loc[B1_cond & ~B2_cond, 'cell_type'] = 'B1'
    adata.obs.loc[B2_cond & ~B1_cond, 'cell_type'] = 'B2'
    adata.obs.loc[B1_cond & B2_cond, 'cell_type'] = 'B1+B2'

    adata = adata[::2].copy() # keep only gene expression

    adata = adata[adata.obs['time'] >= 15.0].copy() # remove early time points where cells haven't fully differentiated yet

    S.compute_fate_probabilities(key=jax.random.PRNGKey(seed+100), 
                                 adata=adata, type_cond_fn=type_cond_fn,
                                 type_label='cell_type', time_col='time', 
                                 max_num_itr=traj_sampling_kwargs['max_num_traj_itr'], 
                                 max_pop_buffer=traj_sampling_kwargs['max_pop_buffer'])

    adata = adata[:,::2].copy() # remove proteins

    # add dropouts
    drop_cutoff = 0.5
    drop_prob = 0.5
    adata = utils.dropout(adata, drop_cutoff, drop_prob, time_key='time')
    # reset cell type annotations after dropout
    adata.obs['cell_type'] = 'Progenitor'
    B1_cond = adata[:, 'tfB11'].X.flatten() > 1.0 # express B1 marker
    B2_cond = adata[:, 'tfB22'].X.flatten() > 1.0 # express B2 marker

    adata.obs.loc[B1_cond & ~B2_cond, 'cell_type'] = 'B1'
    adata.obs.loc[B2_cond & ~B1_cond, 'cell_type'] = 'B2'
    adata.obs.loc[B1_cond & B2_cond, 'cell_type'] = 'B1+B2'


    adata.obs['age'] = adata.obs['time'] 
    del adata.obs['time']

    adata.obsm['X_pca'] = adata.X
    adata.uns['all_cell_types'] = np.unique(adata.obs['cell_type']) 

    sc.pp.neighbors(adata, random_state=seed + 1000)
    sc.tl.umap(adata, random_state=seed + 10000)

    adata.write_h5ad(f'{save_dir}/GRN_simulation_dropout={drop_cutoff}_seed={seed}.h5ad')