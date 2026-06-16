import numpy as np
import pandas as pd
import os
from gstatot import PBA, sweep_utils, utils
import scanpy as sc
import matplotlib.pyplot as plt
import sys
from itertools import product
import sys

data_dir = '../../../../global-stationaryOT/extra/data/hemo_data/prefitted_adatas'
if len(sys.argv) != 5:
    print("Usage: python hemo_sweep_T.py <T> <data_dir> <cell_set> <run_type>")
    exit(1)

T_ind = int(sys.argv[1])
data_dir = str(sys.argv[2])
cell_set = str(sys.argv[3])
run_type = str(sys.argv[4])

save_dir = './sweep_res/integrated_T' + '_' + run_type

n = 100
key = 42 if run_type == 'sweep' else 0 

true_dt = dt = 0.25 
num_experiments = 1 if run_type == 'sweep' else 10

do_prenatal = cell_set == 'prenatal'
do_postnatal = cell_set == 'postnatal'

do_PBA = False 
do_gStatOT = True 
do_StatOT = True 

comp_fp = True
comp_traj = True
comp_prop = False

script_path = '../../../../PBA'
pba_save_dir = './pba_outputs'
py2_path = '../../../../../.pyenv/versions/2.7.18/bin/python2.7'

ks = [3, 5, 7, 10, 15, 20] 
ks = [k for k in ks if k < n] # filter out k values greater than n
Ds = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 5]

lams = [0.0, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0 , 40.0, 50.0, 60.0]
eps2s = [0.04, 0.045, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

sweep_name = run_type + '_' + cell_set

if do_prenatal:
    print("\nSWEEPING PRENATAL DATA:")

    adata_keys  = {'time_key': 'age_wks',
                   'cell_type_key': 'cell_type',
                   'growth_rate_key': 'growth_rate',
                   'embed_key': 'X_pca'}



    adata = sc.read_h5ad(f'{data_dir}/prenatal_integrated_StatOT_fitted.h5ad')

    age_wks = np.unique(adata.obs['age_wks'])
    print(age_wks)
    print('Number of unique time points:', len(age_wks))
    print("\nPrenatal Time point Selection:")
    age_span = np.max(age_wks) - np.min(age_wks)
    print('Age span: ', age_span)


    three_theoretical_time_points = np.linspace(np.min(age_wks), np.max(age_wks), 3, dtype=int)
    print('theoretical three time points:', three_theoretical_time_points)
    three_time_points = np.array([min(age_wks, key=lambda x:abs(x-tp)) for tp in three_theoretical_time_points])
    print('chosen three time points:     ', three_time_points)

    four_theoretical_time_points = np.linspace(np.min(age_wks), np.max(age_wks), 4, dtype=int)
    print('theoretical four time points:', four_theoretical_time_points)
    four_time_points = np.array([min(age_wks, key=lambda x:abs(x-tp)) for tp in four_theoretical_time_points])
    print('chosen four time points:     ', four_time_points)

    five_theoretical_time_points = np.linspace(np.min(age_wks), np.max(age_wks), 5, dtype=int)
    print('theoretical five time points:', five_theoretical_time_points)
    five_time_points = np.array([min(age_wks, key=lambda x:abs(x-tp)) for tp in five_theoretical_time_points])
    print('chosen five time points:     ', five_time_points)

    six_theoretical_time_points = np.linspace(np.min(age_wks), np.max(age_wks), 6, dtype=int)
    print('theoretical six time points:', six_theoretical_time_points)
    six_time_points = np.array([min(age_wks, key=lambda x:abs(x-tp)) for tp in six_theoretical_time_points])
    six_time_points[2] = 14 # correct double use of 15
    print('chosen six time points:     ', six_time_points)

    seven_theoretical_time_points = np.linspace(np.min(age_wks), np.max(age_wks), 7, dtype=int)
    print('theoretical seven time points:', seven_theoretical_time_points)
    seven_time_points = np.array([min(age_wks, key=lambda x:abs(x-tp)) for tp in seven_theoretical_time_points])
    seven_time_points[-2] = 22 # correct double use of 20
    print('chosen seven time points:     ', seven_time_points)

    wk_lists = {
        2: np.array([10.0, 23.0]),
        3: three_time_points,
        4: four_time_points, 
        5: five_time_points, 
        6: six_time_points, 
        7: seven_time_points, 
        8: age_wks # all time points
    }
    wk_keys = list(wk_lists.keys())
    wk_keys.reverse()
    T = wk_keys[T_ind]
    times = wk_lists[T]

    if do_gStatOT:
        adata = sc.read_h5ad(f'{data_dir}/prenatal_integrated_StatOT_fitted.h5ad')
        adata.uns['all_cell_types'] = adata.uns['terminal_lineages']

        adata.uns['true_dt'] = true_dt
        ages = np.unique(adata.obs['age_wks'])
        true_traj = adata.uns['true_traj_data']
        num_step = [true_traj[str(age)].shape[1] for age in times]
        num_traj = true_traj[str(times[0])].shape[0]
        print("Number of trajectories: ", num_traj)
        print("Number of steps: ", num_step)


        gStatOT_params_list = [{'lam': lam, 'epsilon2': eps2, 'dt': dt, 'r' : 1, 'tol': 1e-3, 'solver_type': 'BCA', 
                                'num_step': num_step, 'num_traj': num_traj} for lam, eps2 in product(lams, eps2s)]

        if run_type == 'validation':
            opt_vals = pd.read_csv(f'./gStatOT_opt_vals_prenatal.csv', index_col=0)
            param_dict = {ind : {'lam': opt_vals.loc[ind, 'lam'], 'epsilon2': opt_vals.loc[ind, 'epsilon2'], 
                                 'dt': dt, 'r' : 1, 'tol': 1e-3, 'solver_type': 'BCA', 
                                 'num_step': num_step, 'num_traj': num_traj} for ind in opt_vals.index}
            gStatOT_params_list = [param_dict[T]]

        os.makedirs(save_dir + '/figures/gStatOT', exist_ok=True)


        print(f"Running gStatOT sweep for T={T} time points...")

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
        adata = sc.read_h5ad(f'{data_dir}/prenatal_integrated_StatOT_fitted.h5ad')
        adata.uns['all_cell_types'] = adata.uns['terminal_lineages']
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
            opt_vals = pd.read_csv(f'./StatOT_opt_vals_prenatal.csv', index_col=0)
            param_dict = {ind : {'epsilon': opt_vals.loc[ind, 'epsilon'], 
                                 'dt': dt, 'num_step': num_step, 'num_traj': num_traj} for ind in opt_vals.index}
            StatOT_params_list = [param_dict[T]]

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
        adata = sc.read_h5ad(f'{data_dir}/prenatal_integrated_PBA_fitted.h5ad')
        adata.uns['all_cell_types'] = adata.uns['terminal_lineages']

        adata.uns['true_dt'] = true_dt
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
            opt_vals = pd.read_csv(f'./PBA_opt_vals_prenatal.csv', index_col=0)
            param_dict = {ind : {'k': int(opt_vals.loc[ind, 'k']), 'D': opt_vals.loc[ind, 'D'],
                                  'save_dir': pba_save_dir, 'script_path': script_path, 
                                  'py2_path': py2_path, 'dt' : true_dt, 'use_pca': True,
                                  'num_step': num_step, 'num_traj': num_traj} 
                                  for ind in opt_vals.index}
            PBA_params_list = [param_dict[T]]

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
        fig.savefig(save_dir + f'/figures/PBA/{T}_{sweep_name}_PBA.png')
        plt.close(fig)

if do_postnatal:
    ### POSTNATAL 
    print("\nSWEEEPING POSTNATAL DATA:")
    adata = sc.read_h5ad(f'{data_dir}/postnatal_integrated_StatOT_fitted.h5ad')

    adata_keys  = {'time_key': 'age_yrs',
                   'cell_type_key': 'cell_type',
                   'growth_rate_key': 'growth_rate',
                   'embed_key': 'X_pca'}


    age_yrs = np.unique(adata.obs[adata_keys['time_key']])
    print(len(age_yrs), 'postnatal time points:', age_yrs)
    print('\nPostnatal Time Points Selection:')
    age_span = np.max(age_yrs) - np.min(age_yrs)
    print('Age span: ', age_span)


    four_time_points = np.linspace(np.min(age_yrs), np.max(age_yrs), 4, dtype=int)
    print('theoretical four time points:', four_time_points)
    # get closest actual time points
    four_time_points = np.array([min(age_yrs, key=lambda x:abs(x-tp)) for tp in four_time_points])
    print('chosen four time points:     ', four_time_points)
    print('\n')
    six_time_points = np.linspace(np.min(age_yrs), np.max(age_yrs), 6, dtype=int)
    print('theoretical six time points:', six_time_points)
    six_time_points = np.array([min(age_yrs, key=lambda x:abs(x-tp)) for tp in six_time_points])
    print('chosen six time points:     ', six_time_points)
    print('\n')
    eight_time_points = np.linspace(np.min(age_yrs), np.max(age_yrs), 8, dtype=int)
    print('theoretical eight time points:', eight_time_points)
    eight_time_points = np.array([min(age_yrs, key=lambda x:abs(x-tp)) for tp in eight_time_points])
    print('chosen eight time points:     ', eight_time_points)
    print('\n')
    ten_time_points = np.linspace(np.min(age_yrs), np.max(age_yrs), 10, dtype=int)
    print('theoretical ten time points:', ten_time_points)
    ten_time_points = np.array([min(age_yrs, key=lambda x:abs(x-tp)) for tp in ten_time_points])
    ten_time_points[-2] = 76 # correct double use of 62
    print('chosen ten time points:     ', ten_time_points)
    print('\n')
    twelve_time_points = np.linspace(np.min(age_yrs), np.max(age_yrs), 12, dtype=int)
    print('theoretical twelve time points:', twelve_time_points)
    twelve_time_points = np.array([min(age_yrs, key=lambda x:abs(x-tp)) for tp in twelve_time_points])
    twelve_time_points[-7] = 32 # correct double use of 45
    twelve_time_points[-6] = 35
    print('chosen twelve time points:     ', twelve_time_points)

    year_lists = {
        2 : np.array([min(age_yrs), max(age_yrs)]),
        4 : four_time_points,
        6 : six_time_points,
        8 : eight_time_points,
        10 : ten_time_points,
        12 : twelve_time_points,
        14 : age_yrs # all time points
    }
    yr_keys = list(year_lists.keys())
    yr_keys.reverse()
    T = yr_keys[T_ind] 
    times = year_lists[T]

    if do_gStatOT:

        adata = sc.read_h5ad(f'{data_dir}/postnatal_integrated_StatOT_fitted.h5ad')
        adata.uns['all_cell_types'] = adata.uns['terminal_lineages']
        true_traj = adata.uns['true_traj_data']
        num_step = [true_traj[str(age)].shape[1] for age in times]
        num_traj = true_traj[str(times[0])].shape[0]
        print("Number of trajectories: ", num_traj)
        print("Number of steps: ", num_step)

        gStatOT_params_list = [{'lam': lam, 'epsilon2': eps2, 'dt': dt, 'r' : 1, 'tol': 1e-3, 'solver_type': 'BCA', 
                                'num_step': num_step, 'num_traj': num_traj} for lam, eps2 in product(lams, eps2s)]
        
        if run_type == 'validation':
            opt_vals = pd.read_csv(f'./gStatOT_opt_vals_postnatal.csv', index_col=0)
            param_dict = {ind : {'lam': opt_vals.loc[ind, 'lam'], 'epsilon2': opt_vals.loc[ind, 'epsilon2'], 
                                 'dt': dt, 'r' : 1, 'tol': 1e-3, 'solver_type': 'BCA', 
                                 'num_step': num_step, 'num_traj': num_traj} for ind in opt_vals.index}
            gStatOT_params_list = [param_dict[T]]

        os.makedirs(save_dir + '/figures/gStatOT', exist_ok=True)

        print(f"Running gStatOT sweep for T={T} time points...")

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

        true_traj = adata.uns['true_traj_data']
        ages = np.unique(adata.obs['age_yrs'])
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
            opt_vals = pd.read_csv(f'./StatOT_opt_vals_postnatal.csv', index_col=0)
            param_dict = {ind : {'epsilon': opt_vals.loc[ind, 'epsilon'], 
                                 'dt': dt, 'num_step': num_step, 'num_traj': num_traj} 
                                 for ind in opt_vals.index}

            StatOT_params_list = [param_dict[T]]
        
        
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

        true_traj = adata.uns['true_traj_data']
        ages = np.unique(adata.obs['age_yrs'])
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

            opt_vals = pd.read_csv(f'./PBA_opt_vals_postnatal.csv', index_col=0)
            param_dict = {ind : {'k': int(opt_vals.loc[ind, 'k']), 'D': opt_vals.loc[ind, 'D'], 
                                 'save_dir': pba_save_dir, 'script_path': script_path, 
                                 'py2_path': py2_path, 'dt' : true_dt, 'use_pca': True,
                                 'num_step': num_step, 'num_traj': num_traj} 
                                 for ind in opt_vals.index}
            PBA_params_list = [param_dict[T]]


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
        fig.savefig(save_dir + f'/figures/PBA/{T}_{sweep_name}_PBA.png')
        plt.close(fig)