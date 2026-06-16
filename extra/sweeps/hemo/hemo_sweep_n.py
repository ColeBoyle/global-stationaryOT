import numpy as np
import pandas as pd
import os
from gstatot import sweep_utils 
import scanpy as sc
import matplotlib.pyplot as plt
import sys
from itertools import product


if len(sys.argv) != 6:
    print("Usage: python hemo_sweep_n_prenatal.py <n_ind> <data_dir> <cell_set> <run_type> <w>")
    exit(1)

n_ind = int(sys.argv[1])
data_dir = str(sys.argv[2])
cell_set = str(sys.argv[3]) # 'prenatal' or 'postnatal'
run_type = str(sys.argv[4])
w = int(sys.argv[5])

n_list = [10, 25, 50, 100, 200, 300, 400, 500, 600, 700]
n_list.reverse() # reverse to run larger n values first
n = n_list[n_ind] 

key = 42 if run_type == 'sweep' else 0

true_dt = dt = 0.25
num_experiments = 1 if run_type == 'sweep' else 10
do_prenatal = cell_set == 'prenatal'
do_postnatal = cell_set == 'postnatal'

do_PBA = False
do_gStatOT = True
do_StatOT = False

comp_traj = True
comp_fp = True
comp_prop = False

save_dir = './sweep_res/integrated_n_' + run_type

script_path = '../../../../PBA'
pba_save_dir = './pba_outputs'
py2_path = '../../../../../.pyenv/versions/2.7.18/bin/python2.7'

sweep_name = run_type + '_' + cell_set

lam_w_dict =  {1 : [0, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0],
               10 : [0, 0.1, 0.25, 0.5, 1.0, 2.5],
}
lams = lam_w_dict[w]
eps2_w_dict = {1 : [0.04, 0.045, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
               10 : [0.04, 0.045, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
}
eps2s = eps2_w_dict[w]


ks = [3, 5, 7, 10, 15, 20] 
ks = [k for k in ks if k < n] # filter out k values greater than n
Ds = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 5]

if do_prenatal:
    print("\nSWEEPING PRENATAL DATA:")

    adata_keys  = {'time_key': 'age_wks',
                   'cell_type_key': 'cell_type',
                   'growth_rate_key': 'growth_rate',
                   'embed_key': 'X_pca'}

    # Just load once for common keys
    _tmp_adata = sc.read_h5ad(f'{data_dir}/prenatal_integrated_StatOT_fitted.h5ad')
    age_wks = np.unique(_tmp_adata.obs['age_wks'])
    T = len(age_wks)
    del _tmp_adata

    if do_gStatOT:
        adata = sc.read_h5ad(f'{data_dir}/prenatal_integrated_StatOT_fitted.h5ad')
        adata.uns['all_cell_types'] = adata.uns['terminal_lineages'] 

        adata.uns['true_dt'] = true_dt
        times = age_wks
        true_traj = adata.uns['true_traj_data']
        num_step = [true_traj[str(age)].shape[1] for age in times]
        num_traj = true_traj[str(times[0])].shape[0]
        print("Number of trajectories: ", num_traj)
        print("Number of steps: ", num_step)

        gStatOT_params_list = [{'lam': lam, 'epsilon2': eps2, 'w': w, 'dt': dt, 'r' : 1.0, 
                                'tol': 1e-3, 'solver_type': 'BCA', 'num_step': num_step, 
                                'num_traj': num_traj} for lam, eps2 in product(lams, eps2s)]

        if run_type == 'validation':
            opt_vals = pd.read_csv(f'./gStatOT_opt_vals_prenatal_n.csv', index_col=0)
            param_dict = {ind : {'lam': opt_vals.loc[ind, 'lam'], 'epsilon2': opt_vals.loc[ind, 'epsilon2'], 
                                 'dt': dt, 'r' : 1, 'tol': 1e-3, 'solver_type': 'BCA', 
                                 'num_step': num_step, 'num_traj': num_traj} for ind in opt_vals.index}
            gStatOT_params_list = [param_dict[n]]

        os.makedirs(save_dir + '/figures/gStatOT', exist_ok=True)

        print(f"Running gStatOT sweep for n={n}, T={T} time points...")
        times = age_wks 
        res_df_gStatOT = sweep_utils.sweep_method(method='gStatOT',
                                                  file_name=sweep_name,
                                                  true_adata=adata,
                                                  adata_keys=adata_keys,
                                                  method_params_list=gStatOT_params_list,
                                                  n=n,
                                                  T_time=times,
                                                  seed=key,
                                                  num_exp=num_experiments,
                                                  comp_fp=comp_fp,
                                                  comp_traj=comp_traj,
                                                  comp_prop=comp_prop,
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
        plt.close(fig)

    if do_StatOT:
        print(f"\nRunning StatOT sweep for n={n}, T={T} time points...")
        adata = sc.read_h5ad(f'{data_dir}/prenatal_integrated_StatOT_fitted.h5ad')
        adata.uns['all_cell_types'] = adata.uns['terminal_lineages']

        adata.uns['true_dt'] = true_dt
        times = age_wks
        true_traj = adata.uns['true_traj_data']
        num_step = [true_traj[str(age)].shape[1] for age in times]
        num_traj = true_traj[str(times[0])].shape[0]
        print("Number of trajectories: ", num_traj)
        print("Number of steps: ", num_step)

        os.makedirs(save_dir + '/figures/StatOT', exist_ok=True)
    
        print(f"Running StatOT sweep for n={n}, T={T} time points...")
        StatOT_params_list = [{'epsilon': e, 'dt':dt, 'num_step': num_step, 
                               'num_traj': num_traj} for e in eps2s]
        
        if run_type == 'validation':
            opt_vals = pd.read_csv(f'./StatOT_opt_vals_prenatal_n.csv', index_col=0)
            param_dict = {ind : {'epsilon': opt_vals.loc[ind, 'epsilon'], 
                                 'dt': dt, 'num_step': num_step, 'num_traj': num_traj} for ind in opt_vals.index}
            StatOT_params_list = [param_dict[n]]

        times = age_wks 
        res_df_StatOT = sweep_utils.sweep_method(method='StatOT',
                                                  file_name=sweep_name,
                                                  true_adata=adata,
                                                  adata_keys=adata_keys,
                                                  method_params_list=StatOT_params_list,
                                                  n=n,
                                                  T_time=times,
                                                  seed=key,
                                                  num_exp=num_experiments,
                                                  comp_fp=comp_fp,
                                                  comp_traj=comp_traj,
                                                  comp_prop=comp_prop,
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
        plt.close(fig)

    if do_PBA:
        print(f"\nRunning PBA sweep for n={n}, T={T} time points...")
        os.makedirs(save_dir + '/figures/PBA', exist_ok=True)

        adata = sc.read_h5ad(f'{data_dir}/prenatal_integrated_PBA_fitted.h5ad')
        adata.uns['all_cell_types'] = adata.uns['terminal_lineages']

        adata.uns['true_dt'] = true_dt
        times = age_wks
        true_traj = adata.uns['true_traj_data']
        num_step = [true_traj[str(age)].shape[1] for age in times]
        num_traj = true_traj[str(times[0])].shape[0]
        print("Number of trajectories: ", num_traj)
        print("Number of steps: ", num_step)

        PBA_params_list = [{'k': k, 'D': D,
                            'save_dir': pba_save_dir, 'script_path': script_path, 
                            'py2_path': py2_path, 'dt' : true_dt, 'use_pca': True,
                            'num_step': num_step, 'num_traj': num_traj} for k, D in product(ks, Ds)]
        if run_type == 'validation':
            opt_vals = pd.read_csv(f'./PBA_opt_vals_prenatal_n.csv', index_col=0)
            param_dict = {ind : {'k': int(opt_vals.loc[ind, 'k']), 'D': opt_vals.loc[ind, 'D'],
                                 'save_dir': pba_save_dir, 'script_path': script_path, 
                                 'py2_path': py2_path, 'dt' : true_dt, 'use_pca': True,
                                 'num_step': num_step, 'num_traj': num_traj} for ind in opt_vals.index}
            PBA_params_list = [param_dict[n]]

        # PBA
        times = age_wks
        res_df_PBA = sweep_utils.sweep_method(method='PBA', 
                                              file_name=sweep_name,
                                              true_adata=adata, 
                                              adata_keys=adata_keys, 
                                              method_params_list=PBA_params_list,
                                              n=n,
                                              T_time=times,
                                              seed=key,
                                              num_exp=num_experiments,
                                              comp_fp=comp_fp,
                                              comp_traj=comp_traj,
                                              comp_prop=comp_prop,
                                              save_dir=save_dir
        )

        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        axs = axs.flatten()
        ax1 = sweep_utils.plot_sweep_results(res_df_PBA, 'k', 'D', metric='mean_metric', title='Mean Metric', input_ax=axs[0])
        ax2 = sweep_utils.plot_sweep_results(res_df_PBA, 'k', 'D', metric='marginal_W2_dist', title='Marginal W2 Distance', input_ax=axs[1])
        ax3 = sweep_utils.plot_sweep_results(res_df_PBA, 'k', 'D', metric='FP_TV_dist', title='FP TV Distance', input_ax=axs[2])
        ax4 = sweep_utils.plot_sweep_results(res_df_PBA, 'k', 'D', metric='traj_W2_dist', title='Trajectory W2 Distance', input_ax=axs[3])
        plt.tight_layout()
        fig.savefig(save_dir + f'/figures/PBA/n={n}_T={T}_{sweep_name}_PBA.png')
        plt.close(fig)

if do_postnatal:
    ### POSTNATAL 
    print("\nSWEEEPING POSTNATAL DATA:")


    adata_keys  = {'time_key': 'age_yrs',
                   'cell_type_key': 'cell_type',
                   'growth_rate_key': 'growth_rate',
                   'embed_key': 'X_pca'}

    # Just load once for common keys
    _tmp_adata = sc.read_h5ad(f'{data_dir}/postnatal_integrated_StatOT_fitted.h5ad')
    age_yrs = np.unique(_tmp_adata.obs['age_yrs'])
    T = len(age_yrs)
    del _tmp_adata

    if do_gStatOT:

        adata = sc.read_h5ad(f'{data_dir}/postnatal_integrated_StatOT_fitted.h5ad')

        adata.uns['all_cell_types'] = adata.uns['terminal_lineages']
        
        adata.uns['true_dt'] = true_dt

        times = age_yrs
        true_traj = adata.uns['true_traj_data']
        num_step = [true_traj[str(age)].shape[1] for age in times]
        num_traj = true_traj[str(times[0])].shape[0]
        adata.uns['true_dt'] = true_dt
        print("Number of trajectories: ", num_traj)
        print("Number of steps: ", num_step)

        gStatOT_params_list = [{'lam': lam, 'epsilon2': eps2, 'w':w, 'dt': dt, 'r' : 1.0, 
                                'tol': 1e-3, 'solver_type': 'BCA', 'num_step': num_step, 
                                'num_traj': num_traj} for lam, eps2 in product(lams, eps2s)]

        if run_type == 'validation':
            if n >= 500:
                gStatOT_params_list = [{'lam': 1.0, 'epsilon2': 0.04, 'w': 50,
                                     'dt': dt, 'r' : 1, 'tol': 1e-3, 'solver_type': 'BCA', 
                                     'num_step': num_step, 'num_traj': num_traj}]
            else:
                opt_vals = pd.read_csv(f'./gStatOT_opt_vals_postnatal_n.csv', index_col=0)
                param_dict = {ind : {'lam': opt_vals.loc[ind, 'lam'], 'epsilon2': opt_vals.loc[ind, 'epsilon2'], 
                                     'dt': dt, 'r' : 1, 'tol': 1e-3, 'solver_type': 'BCA', 
                                     'num_step': num_step, 'num_traj': num_traj} for ind in opt_vals.index}
                gStatOT_params_list = [param_dict[n]]


        os.makedirs(save_dir + '/figures/gStatOT', exist_ok=True)

        print(f"Running gStatOT sweep for n={n}, T={T} time points...")
        times = age_yrs

        res_df_gStatOT = sweep_utils.sweep_method(method='gStatOT',
                                                  file_name=sweep_name,
                                                  true_adata=adata,
                                                  adata_keys=adata_keys,
                                                  method_params_list=gStatOT_params_list,
                                                  n=n,
                                                  T_time=times,
                                                  seed=key,
                                                  num_exp=num_experiments,
                                                  comp_fp=comp_fp,
                                                  comp_traj=comp_traj,
                                                  comp_prop=comp_prop,
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
        plt.close(fig)

    if do_StatOT:

        adata = sc.read_h5ad(f'{data_dir}/postnatal_integrated_StatOT_fitted.h5ad')

        adata.uns['all_cell_types'] = adata.uns['terminal_lineages']
        times = age_yrs
        true_traj = adata.uns['true_traj_data']
        num_step = [true_traj[str(age)].shape[1] for age in times]
        num_traj = true_traj[str(times[0])].shape[0]
        adata.uns['true_dt'] = true_dt
        print("Number of trajectories: ", num_traj)
        print("Number of steps: ", num_step)

        os.makedirs(save_dir + '/figures/StatOT', exist_ok=True)
    
        print(f"Running StatOT sweep for T={T} time points...")
        StatOT_params_list = [{'epsilon': e, 'dt':dt, 'num_step': num_step, 
                               'num_traj': num_traj} for e in eps2s]
        if run_type == 'validation':
            opt_vals = pd.read_csv(f'./StatOT_opt_vals_postnatal_n.csv', index_col=0)
            param_dict = {ind : {'epsilon': opt_vals.loc[ind, 'epsilon'], 
                                 'dt': dt, 'num_step': num_step, 'num_traj': num_traj} for ind in opt_vals.index}
            StatOT_params_list = [param_dict[n]]

        times = age_yrs 
        res_df_StatOT = sweep_utils.sweep_method(method='StatOT',
                                                  file_name=sweep_name,
                                                  true_adata=adata,
                                                  adata_keys=adata_keys,
                                                  method_params_list=StatOT_params_list,
                                                  n=n,
                                                  T_time=times,
                                                  seed=key,
                                                  num_exp=num_experiments,
                                                  comp_fp=comp_fp,
                                                  comp_traj=comp_traj,
                                                  comp_prop=comp_prop,
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
        plt.close(fig)

    if do_PBA:

        os.makedirs(save_dir + '/figures/PBA', exist_ok=True)

        adata = sc.read_h5ad(f'{data_dir}/postnatal_integrated_PBA_fitted.h5ad')
        adata.uns['all_cell_types'] = adata.uns['terminal_lineages']


        times = age_yrs
        true_traj = adata.uns['true_traj_data']
        num_step = [true_traj[str(age)].shape[1] for age in times]
        num_traj = true_traj[str(times[0])].shape[0]
        adata.uns['true_dt'] = true_dt

        print("Number of trajectories: ", num_traj)
        print("Number of steps: ", num_step)

        PBA_params_list = [{'k': k, 'D': D,
                            'save_dir': pba_save_dir, 'script_path': script_path, 
                            'py2_path': py2_path, 'dt' : true_dt, 'use_pca': True,
                            'num_step': num_step, 'num_traj': num_traj} for k, D in product(ks, Ds)]

        if run_type == 'validation':
            opt_vals = pd.read_csv(f'./PBA_opt_vals_postnatal_n.csv', index_col=0)
            param_dict = {ind : {'k': int(opt_vals.loc[ind, 'k']), 'D': opt_vals.loc[ind, 'D'],
                                 'save_dir': pba_save_dir, 'script_path': script_path, 
                                 'py2_path': py2_path, 'dt' : true_dt, 'use_pca': True,
                                 'num_step': num_step, 'num_traj': num_traj} for ind in opt_vals.index}
            PBA_params_list = [param_dict[n]]

        times = age_yrs

        res_df_PBA = sweep_utils.sweep_method(method='PBA', 
                                              file_name=sweep_name,
                                              true_adata=adata, 
                                                  adata_keys=adata_keys, 
                                                  method_params_list=PBA_params_list,
                                                  n=n,
                                                  T_time=times,
                                                  seed=key,
                                                  num_exp=num_experiments,
                                                  comp_fp=comp_fp,
                                                  comp_traj=comp_traj,
                                                  comp_prop=comp_prop,
                                                  save_dir=save_dir
            )

        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        axs = axs.flatten()
        ax1 = sweep_utils.plot_sweep_results(res_df_PBA, 'k', 'D', metric='mean_metric', title='Mean Metric', input_ax=axs[0])
        ax2 = sweep_utils.plot_sweep_results(res_df_PBA, 'k', 'D', metric='marginal_W2_dist', title='Marginal W2 Distance', input_ax=axs[1])
        ax3 = sweep_utils.plot_sweep_results(res_df_PBA, 'k', 'D', metric='FP_TV_dist', title='FP TV Distance', input_ax=axs[2])
        ax4 = sweep_utils.plot_sweep_results(res_df_PBA, 'k', 'D', metric='traj_W2_dist', title='Trajectory W2 Distance', input_ax=axs[3])
        plt.tight_layout()
        fig.savefig(save_dir + f'/figures/PBA/n={n}_T={T}_{sweep_name}_PBA.png')
        plt.close(fig)
