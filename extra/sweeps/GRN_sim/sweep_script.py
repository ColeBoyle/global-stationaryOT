import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from gstatot import sweep_utils
from itertools import product

sns.set_style("ticks")
sns.set_context("paper", font_scale=1.5)

data_dir = "../../data/sim_data/BoolODE_GRN_simulations"
save_dir = './sweep_res'

adata = sc.read_h5ad(data_dir + "/GRN_simulation_dropout=0.5_seed=0.h5ad")

sim_dt = dt = adata.uns['true_dt']

sweep_name = f'sweep'

adata_keys = {'time_key': 'age',
              'cell_type_key': 'cell_type',
              'growth_rate_key': 'growth_rate',
              'embed_key': 'X_pca'}

sweep_seed = 0
n = 25

times = np.unique(adata.obs['age'])

lams = [5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 25.0, 30.0 ]
eps2s = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06]

gStatOT_params_list = [{'lam': lam, 'epsilon2': eps2, 'dt': sim_dt, 'r' : 1, 'tol': 1e-3, 
                        'solver_type': 'BCA'} for lam, eps2 in product(lams, eps2s)]

os.makedirs(save_dir + '/figures/gStatOT', exist_ok=True)


res_df_gStatOT = sweep_utils.sweep_method(method='gStatOT',
                                          file_name=sweep_name,
                                          true_adata=adata,
                                          adata_keys=adata_keys,
                                          method_params_list=gStatOT_params_list,
                                          n=n,
                                          T_time=times,
                                          seed=sweep_seed,
                                          num_exp=1,
                                          comp_fp=True,
                                          comp_prop=False,
                                          comp_traj=False,
                                          save_dir=save_dir
)

fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs = axs.flatten()
ax1 = sweep_utils.plot_sweep_results(res_df_gStatOT, 'lam', 'epsilon2', 
                                     metric='mean_metric', title='Mean Metric', input_ax=axs[0])
ax2 = sweep_utils.plot_sweep_results(res_df_gStatOT, 'lam', 'epsilon2', 
                                     metric='marginal_W2_dist', title='Marginal W2 Distance', input_ax=axs[1])
ax3 = sweep_utils.plot_sweep_results(res_df_gStatOT, 'lam', 'epsilon2', 
                                     metric='FP_TV_dist', title='FP TV Distance', input_ax=axs[2])
plt.tight_layout()
fig.savefig(save_dir + f'/figures/gStatOT/{n}_{sweep_name}_gStatOT.png')


# StatOT
eps = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06]
StatOT_params_list = [{'epsilon': e, 'dt':sim_dt} for e in eps]
os.makedirs(save_dir + '/figures/StatOT', exist_ok=True)

res_df_StatOT = sweep_utils.sweep_method(method='StatOT',
                                         file_name=sweep_name,
                                         true_adata=adata,
                                         adata_keys=adata_keys,
                                         method_params_list=StatOT_params_list,
                                         n=n,
                                         T_time=times,
                                         seed=sweep_seed,
                                         num_exp=1,
                                         comp_fp=True,
                                         comp_traj=False,
                                         comp_prop=False,
                                         save_dir=save_dir
)

fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs = axs.flatten()
ax1 = sweep_utils.plot_sweep_results(res_df_StatOT, 'epsilon', None, metric='mean_metric', title='Mean Metric', input_ax=axs[0])
ax2 = sweep_utils.plot_sweep_results(res_df_StatOT, 'epsilon', None, metric='marginal_W2_dist', title='Marginal W2 Distance', input_ax=axs[1])
ax3 = sweep_utils.plot_sweep_results(res_df_StatOT, 'epsilon', None, metric='FP_TV_dist', title='FP TV Distance', input_ax=axs[2])
plt.tight_layout()
fig.savefig(save_dir + f'/figures/StatOT/{n}_{sweep_name}_StatOT.png')

# PBA
ks = [3, 6, 9, 12, 15, 18, 21]
Ds = [0.1, 0.5, 0.75, 1, 5, 10, 20, 40, 50, 60, 80]

script_path = '../../../../PBA'
pba_save_dir = './pba_outputs'
py2_path = '../../../../../.pyenv/versions/2.7.18/bin/python2.7'
os.makedirs(save_dir + '/figures/PBA', exist_ok=True)

PBA_params_list = [{'k': k, 'D': D, 'E': 0.0, 'V': 0.0,
                          'save_dir': pba_save_dir, 'script_path': script_path, 'py2_path': py2_path, 'dt' : sim_dt} for k, D in product(ks, Ds)]

res_df_PBA = sweep_utils.sweep_method(method='PBA', 
                                      file_name=sweep_name,
                                      true_adata=adata, 
                                      adata_keys=adata_keys, 
                                      method_params_list=PBA_params_list,
                                      n=n,
                                      T_time=times,
                                      seed=sweep_seed,
                                      num_exp=1,
                                      comp_fp=True,
                                      comp_traj=False,
                                      comp_prop=False,
                                      save_dir=save_dir
)

fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs = axs.flatten()
ax1 = sweep_utils.plot_sweep_results(res_df_PBA, 'k', 'D', metric='mean_metric', title='Mean Metric', input_ax=axs[0])
ax2 = sweep_utils.plot_sweep_results(res_df_PBA, 'k', 'D', metric='marginal_W2_dist', title='Marginal W2 Distance', input_ax=axs[1])
ax3 = sweep_utils.plot_sweep_results(res_df_PBA, 'k', 'D', metric='FP_TV_dist', title='FP TV Distance', input_ax=axs[2])
plt.tight_layout()
fig.savefig(save_dir + f'/figures/PBA/{n}_{sweep_name}_PBA.png')
