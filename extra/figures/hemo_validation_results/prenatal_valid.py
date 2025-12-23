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
num_step = 65
num_traj = 6000

num_experiments = 10

adata_keys  = {'time_key': 'smoothed_age',
               'cell_type_key': 'sink_type',
               'growth_rate_key': 'growth_rate',
               'embed_key': 'X_pca'}

adata = sc.read_h5ad(f'{data_dir}/prenatal_statOT_fitted.h5ad')

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

gStatOT_params =  {2: {'lam': [5.0], 'epsilon2': [0.03], 'w': [1.0], 'r': [0.1]}, 3: {'lam': [5.0], 'epsilon2': [0.035], 'w': [1.0], 'r': [0.1]}, 4: {'lam': [2.5], 'epsilon2': [0.035], 'w': [1.0], 'r': [0.1]}, 5: {'lam': [2.5], 'epsilon2': [0.03], 'w': [1.0], 'r': [0.1]}, 6: {'lam': [2.5], 'epsilon2': [0.03], 'w': [1.0], 'r': [0.1]}, 7: {'lam': [2.5], 'epsilon2': [0.0275], 'w': [1.0], 'r': [0.1]}, 8: {'lam': [2.5], 'epsilon2': [0.025], 'w': [1.0], 'r': [0.1]}}
StatOT_params =   {2: {'epsilon': [0.035]}, 3: {'epsilon': [0.04]}, 4: {'epsilon': [0.04]}, 5: {'epsilon': [0.04]}, 6: {'epsilon': [0.04]}, 7: {'epsilon': [0.04]}, 8: {'epsilon': [0.04]}}

T = list(wk_lists.keys())[T_ind]
exp_dir = None

all_res_df = pd.DataFrame()
all_res_df = utils.run_sweep(T, wk_lists, n, num_experiments, dt, true_dt, num_step, num_traj, adata, 
                       adata_keys, exp_dir, StatOT_params, gStatOT_params, all_res_df, 
                       max_gStatOT_iter=100_000, max_StatOT_iter=100_000, 
                       key=key, 
                       dtype=jnp.float64, 
                       save_adatas=False,
                       constraint_tol=1e-4,
                       HDR_cutoff=1e-3)

all_res_df.to_csv(f'./prenatal_validation_res_T={T}.csv')