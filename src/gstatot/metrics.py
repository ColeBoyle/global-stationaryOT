import jax.numpy as jnp
import numpy as np
import ot
import pandas as pd
from gstatot import utils
import seaborn as sns
import matplotlib.pyplot as plt
import os

sns.set_context("paper", font_scale=1.5)
sns.set(style="ticks")


class Metric_Evaluator:

    def __init__(self, method, test_adata, true_adata, time_key, 
                 embed_key='X_pca', exp_dir=None, full_supp=True, plot_metrics=True) -> None:

        self.test_adata = test_adata
        self.true_adata = true_adata

        self.time_key = time_key
        self.embed_key = embed_key
        self.method = method
        self.exp_dir = exp_dir
        self.full_supp = full_supp
        self.plot_metrics = plot_metrics

        os.makedirs(self.exp_dir, exist_ok=True) if self.exp_dir is not None else None

        self.results_df = pd.DataFrame(columns=['method', 'Age', 'marginal_W2_dist', 
                                                'traj_W2_dist', 'FP_TV_dist', 'FP_perc_matches'])


        self.times = jnp.array(sorted(self.test_adata.obs[self.time_key].unique()), dtype=jnp.float32)

        self.results_df['Age'] = self.times
        self.T = len(self.times)

        self.results_df['method'] = [self.method] * self.T

    def w2_marginal_error(self, get_dist=True):

        NumItermax = 1_000_000

        w2_dist_list = []
        for i in range(self.T):

            if self.full_supp and get_dist:
                pi_i = self.test_adata.obsp[f'pi_{self.times[i]}']
                test_marginal = np.asarray(pi_i.sum(axis=0), dtype=np.float32)
                test_supp = np.asarray(self.test_adata.obsm[self.embed_key], dtype=np.float32)

            elif self.full_supp and not get_dist:
                test_supp = np.asarray(self.test_adata.obsm[self.embed_key], dtype=np.float32)
                test_marginal = np.ones(test_supp.shape[0], dtype=np.float32) * (1.0 / test_supp.shape[0])

            else:
                test_supp = np.asarray(self.test_adata[self.test_adata.obs[self.time_key] == self.times[i]].obsm[self.embed_key], dtype=np.float32)
                test_marginal = np.ones(test_supp.shape[0], dtype=np.float32) * (1.0 / test_supp.shape[0])


            if jnp.isnan(test_marginal).any():
                print(f"Marginal at time {self.times[i]} contains NaNs, skipping W2 computation")
                w2_dist_list.append(np.nan)
                continue

            true_supp = np.asarray(self.true_adata[self.true_adata.obs[self.time_key] == self.times[i]].obsm['X_pca'], dtype=np.float32)
            true_dist = ot.unif(len(true_supp), type_as=true_supp)

            M_supp = ot.dist(test_supp, true_supp, metric='sqeuclidean')
            W2_dist = np.sqrt(ot.emd2(test_marginal, true_dist, M_supp, numItermax=NumItermax))
            w2_dist_list.append(W2_dist)

        self.results_df['marginal_W2_dist'] = w2_dist_list

        if self.plot_metrics:

            fig, ax = plt.subplots(figsize=(5, 5))
            sns.lineplot(data=self.results_df, x='Age', y='marginal_W2_dist', ax=ax, marker='o')
            ax.grid()
            plt.xlabel('Age')
            plt.ylabel('W2 Distance')
            ax.set_title(f'{self.method}\nW2 marginal error')

            if self.exp_dir is not None:
                fig.savefig(os.path.join(self.exp_dir, f'{self.method}_W2_marginal_error.png'), bbox_inches="tight")
            else:
                plt.show()

            plt.close(fig)


    def w2_trajectory_error(self, test_dt, true_dt, num_traj=None):

        true_dist = [self.true_adata.uns['traj_data'][str(t)] for t in self.times]
        if test_dt >= true_dt:
            true_est_dt_ratio =  int(test_dt / true_dt)
        else:
            true_est_dt_ratio = int(true_dt / test_dt)

        traj_W2_dist = []

        for i in range(self.T):

            traj_data = self.test_adata.uns['traj_data'][str(self.times[i])]

            if np.any(jnp.isnan(traj_data)):
                print(f"Skipping time {i} due to NaN values in trajectory data.")
                traj_W2_dist.append(np.nan)
                continue
            true_traj = true_dist[i]

            if test_dt >= true_dt: # put true traj on test_dt scale
                num_steps = traj_data.shape[1]
                true_traj = true_traj[:, :num_steps*true_est_dt_ratio:true_est_dt_ratio]

            else: # put traj on true_dt scale
                num_steps = true_traj.shape[1]
                traj_data = traj_data[:, :num_steps*true_est_dt_ratio:true_est_dt_ratio]
                true_traj = true_traj[:, :traj_data.shape[1]]

            assert true_traj.shape[1] == traj_data.shape[1], f"Length of true traj: {true_traj.shape[1]} and test traj: {traj_data.shape[1]} do not match."
            cost_mat = np.asarray(utils.compute_traj_cost(true_traj, traj_data), dtype=np.float32)
            unif0 = ot.unif(true_traj.shape[0], type_as=cost_mat)
            unif1 = ot.unif(traj_data.shape[0], type_as=cost_mat)
            W2_cost = np.sqrt(ot.emd2(unif0, unif1, cost_mat, numItermax=1_000_000) )
            traj_W2_dist.append(W2_cost)

        self.results_df['traj_W2_dist'] = traj_W2_dist

        if self.plot_metrics:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(self.times, traj_W2_dist, marker='o')
            ax.axhline(np.mean(traj_W2_dist), linestyle='--', color='red', label='Average W2 distance')
            ax.set_xlabel('Age')
            ax.set_ylabel('W2 distance')
            ax.set_title('W2 distance between true and tests trajectories')
            ax.grid()
    
            if self.exp_dir is not None:
                fig.savefig(os.path.join(self.exp_dir, 'W2_bw_traj.png'), bbox_inches="tight")
            else:
                plt.show()
            plt.close()


    def fp_tv_error(self, label_key, plot_lin_fp=False, true_key=None):
        if true_key is None:
            true_key = f'{label_key}_fp'

        dists = []
        model_FPS = []
        true_FPS = []
        perc_matches = []
        for i in range(self.T):

            sampled_adata_i = self.test_adata[self.test_adata.obs[self.time_key] == self.times[i]]
            sampled_inds = sampled_adata_i.obs.index

            true_FP_i = self.true_adata[sampled_inds].obsm[true_key]

            if self.full_supp:
                model_FP_i = sampled_adata_i.obsm[f'{label_key}_fp_t={self.times[i]}']        
            else:
                model_FP_i = sampled_adata_i.obsm[f'{label_key}_fp']

            if plot_lin_fp:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                ax1.imshow(true_FP_i[:20], aspect='auto')
                ax1.set_title(f'True B_lin at time {self.times[i]}')
                ax2.imshow(model_FP_i[:20], aspect='auto')
                ax2.set_title(f'Model B_lin at time {self.times[i]}')

            TV_distance =jnp.abs(true_FP_i - model_FP_i).sum(axis=1)
            model_FPS.append(model_FP_i)
            true_FPS.append(true_FP_i)
            
            dists.append(np.average(TV_distance))

            # replace nans with -1
            model_FP_i = jnp.where(jnp.isnan(model_FP_i), -1, model_FP_i)
            # see if max fate_prob match
            max_model_prob = jnp.argmax(model_FP_i, axis=1)
            max_true_prob = jnp.argmax(true_FP_i, axis=1)
            num_matches = jnp.sum(max_model_prob == max_true_prob)
            perc_match = num_matches / len(max_model_prob) * 100  
            perc_matches.append(perc_match)


        self.results_df['FP_TV_dist'] = dists
        self.results_df['FP_perc_matches'] = perc_matches

        if self.plot_metrics:
            n_cols = true_FPS[0].shape[1]
            n_rows = len(self.times)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
            for i in range(self.T):
                for j in range(n_cols):
                    ax = axes[i, j]
                    corr = jnp.corrcoef(true_FPS[i][:, j], model_FPS[i][:, j])[0, 1]
                    ax.text(0.1, 0.9, f'Corr: {corr:.2f}', transform=ax.transAxes, fontsize=12, verticalalignment='top')
                    ax.scatter(true_FPS[i][:, j], model_FPS[i][:, j], alpha=0.5)
                    ax.set_title(f'Time {self.times[i]}, Feature {j}')
                    ax.set_xlabel('True B_lin')
                    ax.set_ylabel('Model B_lin')
                    ax.grid()
            plt.tight_layout()
            if self.exp_dir is not None:
                fig.savefig(os.path.join(self.exp_dir, 'lin_fp_corr_scatter.png'))
            plt.close()
    
            fig, (ax1,ax2) = plt.subplots(2,1,figsize=(10, 5), sharex=True)
            ax1.plot(self.times, dists, 'o-')
            ax1.set_ylabel('TV distance')
            ax1.set_title(f'TV distance between true and reconstructed fate probabilities')
            ax1.grid()
    
            ax2.plot(self.times, perc_matches, 'o-')
            ax2.set_xlabel('Age')
            ax2.set_ylabel('Percentage of matches')
            ax2.set_title(f'Percentage of max lineage fate probability matches')
            ax2.grid()
    
            plt.tight_layout()
            if self.exp_dir is not None:
                fig.savefig(os.path.join(self.exp_dir, 'B_lin_TV_distance.png'))
            else:
                plt.show()
            plt.close()
