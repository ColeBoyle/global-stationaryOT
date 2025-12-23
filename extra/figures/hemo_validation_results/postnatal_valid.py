import numpy as np 
import sys
import pandas as pd
import scanpy as sc
import sys
import jax.numpy as jnp
from gstatot import utils

data_dir = '../../data/hemo_data/prefitted_adatas'

T_ind = int(sys.argv[1]) # index of number of time points to use

n = 500
# key = 42 # sweep key
key = 0 # validation key
dt = 0.25
true_dt = 0.25
num_step = 80
num_traj = 8500

num_experiments = 10

adata_keys  = {'time_key': 'smoothed_age',
               'cell_type_key': 'sink_type',
               'growth_rate_key': 'growth_rate',
               'embed_key': 'X_pca'}

adata = sc.read_h5ad(f'{data_dir}/postnatal_statOT_fitted.h5ad')

age_yrs = np.array(sorted(adata.obs[adata_keys['time_key']].unique()))
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

gStatOT_params =  {2: {'lam': [0.0], 'epsilon2': [0.04], 'w': [1.0], 'r': [0.1]}, 4: {'lam': [20.0], 'epsilon2': [0.035], 'w': [1.0], 'r': [0.1]}, 6: {'lam': [15.0], 'epsilon2': [0.035], 'w': [1.0], 'r': [0.1]}, 8: {'lam': [12.0], 'epsilon2': [0.03], 'w': [1.0], 'r': [0.1]}, 10: {'lam': [10.0], 'epsilon2': [0.03], 'w': [1.0], 'r': [0.1]}, 12: {'lam': [10.0], 'epsilon2': [0.03], 'w': [1.0], 'r': [0.1]}, 14: {'lam': [10.0], 'epsilon2': [0.0275], 'w': [1.0], 'r': [0.1]}}
StatOT_params = {2: {'epsilon': [0.04]}, 4: {'epsilon': [0.04]}, 6: {'epsilon': [0.04]}, 8: {'epsilon': [0.04]}, 10: {'epsilon': [0.04]}, 12: {'epsilon': [0.04]}, 14: {'epsilon': [0.04]}}

T = list(year_lists.keys())[T_ind]
exp_dir = None

all_res_df = pd.DataFrame()
all_res_df = utils.run_sweep(T, year_lists, n, num_experiments, dt, true_dt, num_step, num_traj, adata, 
                       adata_keys, exp_dir, StatOT_params, gStatOT_params, all_res_df, 
                       max_gStatOT_iter=100_000, max_StatOT_iter=100_000, 
                       key=key, 
                       dtype=jnp.float64, 
                       save_adatas=False,
                       constraint_tol=1e-4,
                       HDR_cutoff=1e-3)

all_res_df.to_csv(f'./postnatal_valid_results_T={T}.csv')