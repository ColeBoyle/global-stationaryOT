import jax
import numpy as np
from matplotlib import pyplot as plt
import scanpy as sc
import seaborn as sns
from itertools import product
from gstatot import sweep_utils
import os
sns.set_context("paper", font_scale=1.5)
sns.set_theme(style="ticks")
import sys
# get T from arguments

if len(sys.argv) != 3:
    print("Usage: python sim_sweep_script.py <T> <data_dir>")
    exit(1)

T_ind = int(sys.argv[1])
data_dir = str(sys.argv[2])

# load sim adata with ground truth trajectories and fate probability estimates
sim_adata = sc.read_h5ad(f"{data_dir}/bistable_sim_seed=0.h5ad") 

gStatOT = False
StatOT = False
PBA = True

# set time points for time courses
first_time = 2
final_time = 100

sampling_rates = [
    98, #  (day 2 to day 100)
    30, # (monthly)
    14, # (biweekly)
    10, # (every 10 days)
    7, # (every week)
    5, # (every 5 days)
    4, # (every 4 days)
    3, # (every 3 days)
    2, # (every 2 days) 
]

time_lists = {}
for sr in sampling_rates:
    # select time points starting at day 2 then every sr days until day 100
    times = np.arange(first_time, final_time + 1, sr)
    time_lists[len(times)] = times

T = list(time_lists.keys())[T_ind] # select T based on input argument 

# trajectory sampling parameters
sim_dt = dt =  0.01
num_step = int(2/dt) # 2 day long trajectories
num_traj = sim_adata.uns['true_traj_data']['0.0'].shape[0] # 500 
n = 25
sweep_seed = 0

adata_keys  = {'time_key': 'chronological_age', # key in adata.obs for age annotation
               'cell_type_key': 'cell_type', # key in adata.obs for cell type annotation
               'growth_rate_key': 'growth_rate', # key in adata.obs for cell growth rates
               'embed_key': 'X_pca'}

sim_adata.uns['model_dt'] = sim_dt

save_dir = f'./sweep_res'
os.makedirs(save_dir, exist_ok=True)

sweep_name = 'sim'

print("Running sweeps for simulated data with T =", T, "time points and n =", n, "cells...")

##################### gStatOT Sweep #####################
if gStatOT:
    lams = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]
    eps2s = [0.005, 0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2]

    gStatOT_params_list = [{'lam': lam, 'epsilon2': eps2, 'dt': sim_dt, 'r' : 0.1, 'tol': 1e-3, 
                            'solver_type': 'BCA', 'num_step': num_step, 'num_traj': num_traj} for lam, eps2 in product(lams, eps2s)]

    os.makedirs(save_dir + '/figures/gStatOT', exist_ok=True)


    print(f"Running gStatOT sweep for T={T} time points...")
    times = time_lists[T]
    res_df_gStatOT = sweep_utils.sweep_method(method='gStatOT',
                                              file_name=sweep_name,
                                              true_adata=sim_adata,
                                              adata_keys=adata_keys,
                                              method_params_list=gStatOT_params_list,
                                              n=n,
                                              T_time=times,
                                              seed=sweep_seed,
                                              num_exp=1,
                                              comp_fp=True,
                                              comp_traj=True,
                                              comp_prop=False,
                                              save_dir=save_dir
    )

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs = axs.flatten()
    ax1 = sweep_utils.plot_sweep_results(res_df_gStatOT, 'lam', 'epsilon2', 
                                         metric='mean_metric', title='Mean Metric', input_ax=axs[0])
    ax2 = sweep_utils.plot_sweep_results(res_df_gStatOT, 'lam', 'epsilon2', 
                                         metric='marginal_W2_dist', title='Marginal W2 Distance', input_ax=axs[1])
    ax3 = sweep_utils.plot_sweep_results(res_df_gStatOT, 'lam', 'epsilon2', 
                                         metric='FP_TV_dist', title='FP TV Distance', input_ax=axs[2])
    ax4 = sweep_utils.plot_sweep_results(res_df_gStatOT, 'lam', 'epsilon2',
                                         metric='traj_W2_dist', title='Trajectory W2 Distance', input_ax=axs[3])
    plt.tight_layout()
    fig.savefig(save_dir + f'/figures/gStatOT/n={n}_T={T}_{sweep_name}_gStatOT.png')




###################### StatOT Sweep ######################

if StatOT:
    os.makedirs(save_dir + '/figures/StatOT', exist_ok=True)
    
    
    print(f"Running StatOT sweep for T={T} time points...")
    StatOT_params_list = [{'epsilon': e, 'dt':sim_dt, 'num_step': num_step, 'num_traj': num_traj} for e in eps2s]
    times = time_lists[T]
    res_df_StatOT = sweep_utils.sweep_method(method='StatOT',
                                              file_name=sweep_name,
                                              true_adata=sim_adata,
                                              adata_keys=adata_keys,
                                              method_params_list=StatOT_params_list,
                                              n=n,
                                              T_time=times,
                                              seed=sweep_seed,
                                              num_exp=1,
                                              comp_fp=True,
                                              comp_traj=True,
                                              comp_prop=False,
                                              save_dir=save_dir
    )
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs = axs.flatten()
    ax1 = sweep_utils.plot_sweep_results(res_df_StatOT, None, 'epsilon', 
                                         metric='mean_metric', title='Mean Metric', input_ax=axs[0])
    ax2 = sweep_utils.plot_sweep_results(res_df_StatOT, None, 'epsilon', 
                                         metric='marginal_W2_dist', title='Marginal W2 Distance', input_ax=axs[1])
    ax3 = sweep_utils.plot_sweep_results(res_df_StatOT, None, 'epsilon', 
                                         metric='FP_TV_dist', title='FP TV Distance', input_ax=axs[2])
    ax4 = sweep_utils.plot_sweep_results(res_df_StatOT, None, 'epsilon',
    
                                         metric='traj_W2_dist', title='Trajectory W2 Distance', input_ax=axs[3])
    plt.tight_layout()
    fig.savefig(save_dir + f'/figures/StatOT/n={n}_T={T}_{sweep_name}_StatOT.png')



##################### PBA Sweep #####################

if PBA:

    os.makedirs(save_dir + '/figures/PBA', exist_ok=True)
    ks = [3, 6, 9, 12, 15, 18, 21, 24]
    Ds = [0.5, 0.75, 1, 5, 10, 50, 100, 250, 500]

    script_path = '../../../../PBA'
    pba_save_dir = './pba_outputs'
    py2_path = '../../../../../.pyenv/versions/2.7.18/bin/python2.7'

    PBA_params_list = [{'k': k, 'D': D, 
                        'save_dir': pba_save_dir, 'script_path': script_path, 'py2_path': py2_path,
                        'num_step': num_step, 'num_traj': num_traj, 'dt': sim_dt} for i, (k, D) in enumerate(product(ks, Ds), 1)]

    print(f"Running PBA sweep for T={T} time points...")
    times = time_lists[T]
    res_df_PBA = sweep_utils.sweep_method(method='PBA',
                                              file_name=sweep_name,
                                              true_adata=sim_adata,
                                              adata_keys=adata_keys,
                                              method_params_list=PBA_params_list,
                                              n=n,
                                              T_time=times,
                                              seed=sweep_seed,
                                              num_exp=1,
                                              comp_fp=True,
                                              comp_traj=True,
                                              comp_prop=False,
                                              save_dir=save_dir
    )

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs = axs.flatten()
    ax1 = sweep_utils.plot_sweep_results(res_df_PBA, 'k', 'D', 
                                         metric='mean_metric', title='Mean Metric', input_ax=axs[0])
    ax2 = sweep_utils.plot_sweep_results(res_df_PBA, 'k', 'D', 
                                         metric='marginal_W2_dist', title='Marginal W2 Distance', input_ax=axs[1])
    ax3 = sweep_utils.plot_sweep_results(res_df_PBA, 'k', 'D', 
                                         metric='FP_TV_dist', title='FP TV Distance', input_ax=axs[2])
    ax4 = sweep_utils.plot_sweep_results(res_df_PBA, 'k', 'D',
                                         metric='traj_W2_dist', title='Trajectory W2 Distance', input_ax=axs[3])
    plt.tight_layout()
    fig.savefig(save_dir + f'/figures/PBA/n={n}_T={T}_{sweep_name}_PBA.png')