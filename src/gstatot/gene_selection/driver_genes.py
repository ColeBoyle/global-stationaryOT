import os
from tabnanny import verbose
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from tqdm import tqdm

from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from scipy.interpolate import make_splrep
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad

sns.set_style("ticks")
sns.set_palette("deep")

class gene_selection:
    def __init__(self, adata, adata_keys, fate_names=None, full_supp=True):

        for key in adata_keys.keys():
            setattr(self, key, adata_keys[key])

        if 'time_key' not in adata_keys.keys():
            print("Must provide time_key in adata_keys")
            return

        self.full_supp = full_supp
        self.adata = adata

        if fate_names is not None:
            self.fate_names = fate_names
        else:
            self.fate_names = np.unique(self.adata.obs[self.cell_type_key])


        metrics = ['sobolev_norm', 'spearmanr_corr']

        self.times = np.unique(self.adata.obs[self.time_key])
        self.results_df = xr.DataArray(
            np.zeros((len(self.adata.var.index), len(self.fate_names), len(metrics))),
            dims=["gene", "fate", "metric"],
            coords={"gene": self.adata.var.index, "fate": self.fate_names, "metric": metrics}
        ) 
        cmap = plt.get_cmap('tab20')
        adata.var['colour'] = [cmap(i % 20) for i in range(len(adata.var.index))]

        self.genes_idx = None

    def get_fp_expression_corr(self, label_key=None, fate_names=None, genes='all', bootstrap=False, n_bootstraps=100, batch_size=500):

        if fate_names is None:
            fate_names = self.fate_names

        if label_key is None:
            label_key = self.cell_type_key

        if genes == 'all':
            self.genes_idx = self.adata.var.index

        elif genes == 'TF':
            self.genes_idx = self.adata.var.loc[self.adata.var['TF']].index
            
        else:
            self.genes_idx = genes

        def corr_coef(x, y, w):
            w = w / jnp.sum(w)
            mx = jnp.sum(w * x)
            my = jnp.sum(w * y)
            cov = jnp.sum(w * (x - mx) * (y - my))
            sx = jnp.sqrt(jnp.sum(w * (x - mx) ** 2))
            sy = jnp.sqrt(jnp.sum(w * (y - my) ** 2))
            corr = cov / (sx * sy) # will return nan if sx or sy is 0; as desired 
            return corr 

        corr_coef_vmap = jax.jit(jax.vmap(jax.vmap(corr_coef, in_axes=(None, 0, None)), in_axes=(0, None, None)))

        def corr_coef_vmap_batched(x, y, w, batch_size=500):
            corrs = []
            for i in range(0, x.shape[0], batch_size):
                corrs.append(corr_coef_vmap(x[i:i+batch_size], y, w))
            return jnp.concatenate(corrs, axis=0)

        def corr_coef_bootstrap(x, y, w, n_bootstraps, batch_size=500):
            def single_bootstrap(seed):
                idx = jax.random.choice(seed, x.shape[1], shape=(x.shape[1],), replace=True, p=w/w.sum())
                return corr_coef_vmap_batched(x[:, idx], y[:, idx], jnp.ones(x.shape[1]), batch_size=batch_size)

            seeds = jax.random.split(jax.random.PRNGKey(0), n_bootstraps)
            corrs = []
            for seed in seeds:
                corrs.append(single_bootstrap(seed))
            return jnp.stack(corrs)

        self.gene_corrs = []
        if bootstrap:
            print("Calculating bootstrap confidence intervals for gene correlations...")
            df_all = pd.DataFrame()

        for age in tqdm(self.times):
            if self.full_supp:
                adata_a = self.adata[:, self.genes_idx]
                expr = jnp.asarray(adata_a.X.toarray(), dtype=jnp.float32)
                fp = jnp.asarray(adata_a.obsm[f'{label_key}_fp_t={age}'], dtype=jnp.float32)
                weights = jnp.asarray(adata_a.obsp[f'pi_{age}'], dtype=jnp.float32).sum(axis=0)
                # remove nan rows from fp (occurs when fp calculated with HDR_cutoff > 0)
                valid_idx = ~jnp.isnan(fp).any(axis=1)
                expr = expr[valid_idx]
                fp = fp[valid_idx]
                weights = weights[valid_idx].flatten()

                if bootstrap:
                    corrs = corr_coef_bootstrap(expr.T, fp.T, weights, n_bootstraps=n_bootstraps, batch_size=batch_size) # shape (n_bootstraps, n_genes, n_fates)
                    corrs = jnp.where((corrs <= -1) | (corrs >= 1), jnp.nan, corrs)
                    mean_corrs = jnp.mean(corrs, axis=0) # shape (n_genes, n_fates)
                    low_ci = jnp.percentile(corrs, 2.5, axis=0)  # shape (n_genes, n_fates)
                    high_ci = jnp.percentile(corrs, 97.5, axis=0) # shape (n_genes, n_fates)
                    for j in range(corrs.shape[2]):
                        df = pd.DataFrame({
                            'gene': adata_a.var.index,
                            'mean_corr': mean_corrs[:, j],
                            'low_ci': low_ci[:, j],
                            'high_ci': high_ci[:, j],
                            'fate': fate_names[j],
                            'age': age
                        })
                        df_all = pd.concat([df_all, df], ignore_index=True)
                else:
                    corrs = corr_coef_vmap_batched(expr.T, fp.T, weights, batch_size=batch_size)
                    # set values outside -1 to 1 range to nan
                    corrs = jnp.where((corrs <= -1) | (corrs >= 1), jnp.nan, corrs)
                    corr_df = pd.DataFrame(corrs, index=adata_a.var.index, columns=fate_names)
                    self.gene_corrs.append(corr_df)
                    

            else:
                adata_a = self.adata[self.adata.obs[self.time_key] == age, self.genes_idx]   
                expr = jnp.asarray(adata_a.X.toarray(), dtype=jnp.float32)
                fp = jnp.asarray(adata_a.obsm[f'{label_key}_fp'], dtype=jnp.float32)
                weights = jnp.ones((expr.shape[0],), dtype=jnp.float32)

                # remove nan rows from fp and weights
                valid_idx = ~jnp.isnan(fp).any(axis=1)
                expr = expr[valid_idx]
                fp = fp[valid_idx]
                weights = weights[valid_idx].flatten()

                if bootstrap:
                    corrs = corr_coef_bootstrap(expr.T, fp.T, weights, n_bootstraps=n_bootstraps, batch_size=batch_size) # shape (n_bootstraps, n_genes, n_fates)
                    corrs = jnp.where(jnp.isnan(corrs), 0, corrs) # set nan values to 0 for mean and ci calculations; we will set final values to nan later
                    corrs = jnp.where((corrs <= -1) | (corrs >= 1), 0, corrs)
                    mean_corrs = jnp.mean(corrs, axis=0)
                    low_ci = jnp.percentile(corrs, 2.5, axis=0)
                    high_ci = jnp.percentile(corrs, 97.5, axis=0)
                    
                    for j in range(corrs.shape[2]):
                        df = pd.DataFrame({
                            'gene': adata_a.var.index,
                            'mean_corr': mean_corrs[:, j],
                            'low_ci': low_ci[:, j],
                            'high_ci': high_ci[:, j],
                            'fate': fate_names[j],
                            'age': age
                        })
                        df_all = pd.concat([df_all, df], ignore_index=True)
                
                else:
                    corrs = corr_coef_vmap_batched(expr.T, fp.T, weights, batch_size=batch_size)
                    # set values outside -1 to 1 range to nan
                    corrs = jnp.where((corrs <= -1) | (corrs >= 1), jnp.nan, corrs)
                    corr_df = pd.DataFrame(corrs, index=adata_a.var.index, columns=fate_names)
                    self.gene_corrs.append(corr_df)

        # Format data in xarray
        if bootstrap:
            self.gene_corrs_df = df_all                   
        else:
            self.gene_corrs_xr = xr.DataArray(
                np.stack([df.values for df in self.gene_corrs]),
                dims=["age", "gene", "fate"],
                coords={"age": self.times, "gene": self.genes_idx, "fate": fate_names}
            )

    def smooth_gene_corrs(self, k=3, s=None, L2_weighted=False, equal_spacing=True, genes='all'):
        
        if self.genes_idx is None:
            print("Must run get_fp_expression_corr before smoothing gene correlations")
            return
        if not equal_spacing:
            fit_times = (self.times - self.times.min()) / (self.times.max() - self.times.min())
        else:
            fit_times = np.array(list(range(len(self.times))))

        sample_times = np.linspace(fit_times.min(), fit_times.max(), 100)

        self.smoothed_gene_corrs_xr = xr.DataArray(
            np.zeros((len(sample_times), len(self.genes_idx), len(self.fate_names))),
            dims=["time", "gene", "fate"],
            coords={"time": sample_times, "gene": self.genes_idx, "fate": self.fate_names}
        )
        self.smoothed_gene_corrs_splines = xr.DataArray(
            np.empty((len(self.genes_idx), len(self.fate_names)), dtype=object),
            dims=["gene", "fate"],
            coords={"gene": self.genes_idx, "fate": self.fate_names}
        )

        for gene in tqdm(self.gene_corrs_xr.gene.values):
            for fate in self.gene_corrs_xr.fate.values:

                y = self.gene_corrs_xr.sel(gene=gene, fate=fate).values
                ft = fit_times[~np.isnan(y)]  # remove corresponding times
                y = y[~np.isnan(y)]  # remove nan values

                if len(y) < k + 1:
                    self.smoothed_gene_corrs_splines.loc[gene, fate] = None
                    self.smoothed_gene_corrs_xr.loc[:, gene, fate] = np.nan 
                    self.results_df.loc[gene, fate, 'sobolev_norm'] = np.nan
                    continue

                if s is None:
                    B_spline =  UnivariateSpline(ft, y, k=k)
                else:
                    B_spline = make_splrep(ft, y, k=k, s=s)

                self.smoothed_gene_corrs_splines.loc[gene, fate] = B_spline
                self.smoothed_gene_corrs_xr.loc[:, gene, fate] = B_spline(sample_times)

                sobolev_norm = self.compute_sobolev_norm(B_spline, sample_times[0], sample_times[-1], L2_weighted=L2_weighted, full_norm=True)
                self.results_df.loc[gene, fate, 'sobolev_norm'] = sobolev_norm 
                spearmanr_corr = pd.Series(y).corr(pd.Series(ft), method='spearman')
                self.results_df.loc[gene, fate, 'spearmanr_corr'] = spearmanr_corr
    


    def compute_sobolev_norm(self, B_spline, t0, t1, L2_weighted=False, full_norm=False):

        if full_norm:
            sobolev_norm = quad(lambda x: B_spline(x)**2 + B_spline.derivative()(x)**2, t0, t1)[0]**(1/2) / (t1 - t0)

        else:
            sobolev_norm = quad(lambda x: B_spline.derivative()(x)**2, t0, t1)[0]**(1/2) / (t1 - t0)

        if L2_weighted:
            L2_norm, _ = quad(lambda x: B_spline(x)**2, t0, t1) / (t1 - t0)

            return np.sqrt(sobolev_norm * L2_norm)
        else:
            return np.sqrt(sobolev_norm)


    def rank_genes(self, n_top_genes, method='max_corr', genes='all', use_abs=False):

        top_genes_by_fate = {}
        if genes == 'all':
            genes = self.adata.var.index

        elif genes == 'TF':
            genes = self.adata.var.loc[self.adata.var['TF']].index

        if method == 'max_corr':

            for fate in self.fate_names:
                top_genes_by_fate[fate] = {}
                # find top n genes at each age
                for age in self.times:
                    driver_genes_a = self.gene_corrs_xr.sel(age=age, gene=genes, fate=fate).to_series()
                    if use_abs:
                        driver_genes_a = driver_genes_a.abs()
                    top_genes = driver_genes_a.sort_values(ascending=False).head(n_top_genes).index

                    top_genes_by_fate[fate][age] = pd.DataFrame(index=list(top_genes), columns=['corr'])
                    top_genes_by_fate[fate][age]['corr'] = driver_genes_a.loc[list(top_genes)]


            self.cell_type_top_genes = top_genes_by_fate
 
        else:
            raise ValueError(f"Unknown method: {method}")

        return top_genes_by_fate

    def plot_top_corr_over_age(self, fate_names, n_top_genes=10, cell_types=None, only_TF=False, rank_by_abs_corr=False, smoothed=False, plot=True, save_fig=False, fig_dir=None):
        cell_type_top_genes = {}

        for fate in fate_names:
            top_genes = set()
            # find top n genes at each age
            for driver_genes_a in self.gene_corrs:
                if only_TF:
                    driver_genes_a = driver_genes_a.loc[self.adata[:, self.adata.var['TF']].var.index]

                if rank_by_abs_corr:
                    top_genes = top_genes.union(set(driver_genes_a[fate].abs().sort_values(ascending=False).head(n_top_genes).index))

                else:
                    top_genes = top_genes.union(set(driver_genes_a[fate].sort_values(ascending=False).head(n_top_genes).index))

            cell_type_top_genes[fate] = pd.DataFrame(index=list(top_genes))

            for i, driver_genes_a in enumerate(self.gene_corrs):
                cell_type_top_genes[fate]['corr_age_' + str(self.times[i])] = driver_genes_a.loc[list(top_genes)][fate]
            
        self.cell_type_top_genes = cell_type_top_genes
        all_celltypes = cell_types if cell_types is not None else fate_names

        if plot:
            for fate in all_celltypes:
                genes = list(cell_type_top_genes[fate].index)
                if smoothed:
                    gene_corr_over_age = self.smoothed_gene_corrs_xr.sel(gene=genes, fate=fate).values.T
                    t = self.smoothed_gene_corrs_xr.coords['time'].values
                    sobolev_norms = self.results_df.loc[genes, fate, 'sobolev_norm'].values
                    marker = ''
                else:
                    gene_corr_over_age = self.gene_corrs_xr.sel(gene=genes, fate=fate).values.T 
                    t = self.gene_corrs_xr.coords['age'].values
#                    sobolev_norms = self.results_df.loc[genes, fate, 'sobolev_norm'].values
                    marker = 'o'
    
                fig, ax1 = plt.subplots( figsize=(6, 6))
    
                for i, gene in enumerate(genes):
                    # check if var dataframe has gene_name column
                    if self.adata.var.columns.isin(['gene_name']).any():
                        gene_name = self.adata.var.loc[gene]['gene_name']
                    else:
                        gene_name = gene
                    ax1.plot(t, gene_corr_over_age[i], label=gene_name, marker=marker, color=self.adata.var.loc[gene]['colour'])
    
                ax1.set_xlabel('Age')
                ax1.set_ylabel('Correlation b/w gene expression and fate probability')
                if only_TF:
                    ax1.set_title(f'Top TF driver genes for {fate}')
                else:
                    ax1.set_title(f'Top driver genes for {fate}')
               
                ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax1.grid()
                if save_fig:
                    if fig_dir is None:
                        fig_dir = './figs'
                    os.makedirs(fig_dir, exist_ok=True)
                    plt.savefig(f"{fig_dir}/top_{n_top_genes}_driver_genes_{fate}.png", dpi=300, bbox_inches='tight')
                else:
                    plt.show()


    def plot_genes(self, genes, fate, smoothed=True, equal_spacing=False):
        if smoothed:
            gene_corr_over_age = self.smoothed_gene_corrs_xr.sel(gene=genes, fate=fate).values.T
            t = self.smoothed_gene_corrs_xr.coords['time'].values
            if equal_spacing:
                t = np.arange(0, len(t))
            sobolev_norms = self.results_df.loc[genes, fate, 'sobolev_norm'].values
            spearmanr_corrs = self.results_df.loc[genes, fate, 'spearmanr_corr'].values
            marker = ''
        else:
            gene_corr_over_age = self.gene_corrs_xr.sel(gene=genes, fate=fate).values.T 

            t = self.gene_corrs_xr.coords['age'].values
            if equal_spacing:
                t = np.arange(0, len(t))
            marker = 'o'
            sobolev_norms = self.results_df.loc[genes, fate, 'sobolev_norm'].values
            spearmanr_corrs = self.results_df.loc[genes, fate, 'spearmanr_corr'].values

        fig, ax1 = plt.subplots( figsize=(6, 6))
        for i, gene in enumerate(genes):
            # check if var dataframe has gene_name column
            if self.adata.var.columns.isin(['gene_name']).any():
                gene_name = self.adata.var.loc[gene]['gene_name']
            else:
                gene_name = gene
            ax1.plot(t, gene_corr_over_age[i], label=gene_name + f' (sobolev norm: {sobolev_norms[i]:.2e},Spearman: {spearmanr_corrs[i]:.2f})', 
                     marker=marker, color=self.adata.var.loc[gene]['colour'])

        ax1.set_xlabel('Age')
        ax1.set_ylabel('Correlation b/w gene expression and fate probability')
        ax1.set_title(f'Correlation trends for {fate}')
    
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid()
        plt.show()
 


    # select driver genes with lasso and elastic net regression, using gene expression to predict fate probability at each age.

    def select_driver_genes_sparse_regression(self, adata, label_key, fate_name, genes='all', verbose=False, model_type='lasso', alpha=0.001):
        if genes == 'all':
            genes_idx = adata.var.index
            
        elif genes == 'TF':
            genes_idx = adata.var.loc[adata.var['TF']].index

        else:
            genes_idx = genes

        results_df = {'age': [], 'gene_id': [], 'coeff': [], 'test_score': []} 
        for age in tqdm(self.times):

            unique_labels = np.unique(adata.obs[label_key])

            if self.full_supp:
                weights = adata.obsp[f'pi_{age}'].sum(axis=1) 
                X = adata[:, genes_idx].X

                if not isinstance(X, np.ndarray):
                    X = X.toarray()
                
                y = adata.obsm[f'{label_key}_fp_t={age}'][:, np.where(unique_labels == fate_name)[0][0]]

            else:
                adata_a = adata[adata.obs[self.time_key] == age, genes_idx]
                X = adata_a.X
                if not isinstance(X, np.ndarray):
                    X = X.toarray()
                y = adata_a.obsm[f'{label_key}_fp'][:, np.where(unique_labels == fate_name)[0][0]]
                # remove nan values from y and corresponding rows from X 
                weights = np.ones((X.shape[0],))

#            valid_idx = ~np.isnan(y)
#            X = X[valid_idx]
#            y = y[valid_idx]
            if len(y) == 0:
                if verbose:
                    print(f'No valid samples for {fate_name} at age {age}, skipping...')
                continue

            X_train, X_test, y_train, y_test, w_train, w_test = X, X, y,y, weights, weights#train_test_split(X, y, weights, test_size=0.1, random_state=42)

            from sklearn.linear_model import Lasso
            if model_type == 'lasso':
                model = make_pipeline(StandardScaler(), Lasso(alpha=alpha))
            elif model_type == 'lassocv':
                model = make_pipeline(StandardScaler(), LassoCV(cv=5))
            elif model_type == 'elasticnetcv':
                model = make_pipeline(StandardScaler(), ElasticNetCV(cv=5, l1_ratio=0.9))
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            model.fit(X_train, y_train, **{f'{model_type}__sample_weight': w_train})

            coef = model.named_steps[model_type].coef_

            final_score = model.score(X_test, y_test, sample_weight=w_test)

            results_df['age'].extend([age] * coef.shape[0])
            results_df['gene_id'].extend(genes_idx)
            results_df['coeff'].extend(coef)
            results_df['test_score'].extend([final_score] * coef.shape[0])

        results_df = pd.DataFrame(results_df)
        if 'gene_name' in adata.var.columns:
            results_df['gene_name'] = results_df['gene_id'].map(adata.var['gene_name'])

        results_df.loc[:, 'mean_coef'] = results_df.groupby(['gene_id', 'gene_name'])['coeff'].transform('mean')

        results_df = results_df.sort_values('mean_coef', ascending=False)
#        results_df['gene_colour'] = results_df['gene_id'].map(adata.var['colour'])

        return results_df
    
    def select_driver_genes(self, alpha_list, adata, thresh, label_key, fate_name, genes='all', verbose=False, model_type='lasso',
                            sample_fraction=0.5, n_bootstraps=100):
        if genes == 'all':
            genes_idx = adata.var.index
            
        elif genes == 'TF':
            genes_idx = adata.var.loc[adata.var['TF']].index

        else:
            genes_idx = genes

        results = {'age': [], 'sel_genes': [], 'freq': [], 'coeffs': []} 

        for age in tqdm(self.times):
            unique_labels = np.unique(adata.obs[label_key])

            if self.full_supp:
                weights = adata.obsp[f'pi_{age}'].sum(axis=1) 
                X = adata[:, genes_idx].X

                if not isinstance(X, np.ndarray):
                    X = X.toarray()
                
                y = adata.obsm[f'{label_key}_fp_t={age}'][:, np.where(unique_labels == fate_name)[0][0]]

            else:
                adata_a = adata[adata.obs[self.time_key] == age, genes_idx]
                X = adata_a.X
                if not isinstance(X, np.ndarray):
                    X = X.toarray()
                y = adata_a.obsm[f'{label_key}_fp'][:, np.where(unique_labels == fate_name)[0][0]]
                # remove nan values from y and corresponding rows from X 
                weights = np.ones((X.shape[0],))

         
            if len(y) == 0:
                if verbose:
                    print(f'No valid samples for {fate_name} at age {age}, skipping...')
                continue

#            elif model_type == 'lasso':
           
#            elif model_type == 'elasticnetpath':
#                from sklearn.linear_model import enet_path
#                def manual_pipline(X, y, weights):
#                    X_scaled = StandardScaler().fit_transform(X)
#                    alphas_enet, coefs_enet, _ = enet_path(X_scaled, y, l1_ratio=0.9, n_alphas=5)
#                    return alphas_enet, coefs_enet
#                model = manual_pipline

            selection_freq = np.zeros((X.shape[1], len(alpha_list)))

#            from sklearn.linear_model import Lasso
            from sklearn.linear_model import ElasticNet
            from sklearn.linear_model._coordinate_descent import _alpha_grid
            scaler = StandardScaler() 
            X, y, w = scaler.fit_transform(X), y, weights

            alpha_grid = _alpha_grid(X, y, l1_ratio=0.9, n_alphas=5)
            print("Using alpha grid: ", alpha_grid)

            for alpha in alpha_list:

                model = ElasticNet(alpha=alpha, l1_ratio=0.9)
 
                selection_freq_alpha, coef_array, _ = stability_selection(model, X, y, weights, 
                                                                           n_bootstraps=n_bootstraps, 
                                                                           sample_fraction=sample_fraction, 
                                                                           random_state=42,
                                                                           model_type='elasticnet')
                
                selection_freq[:, alpha_list.index(alpha)] = selection_freq_alpha
                print('Completed stability selection for alpha = ', alpha)

            sel_genes = genes_idx[selection_freq.max(axis=1) >= thresh]

            from sklearn.linear_model import LinearRegression
            age_model = make_pipeline(StandardScaler(), LinearRegression())
            X_selected = X[:, np.isin(genes_idx, sel_genes)]
            age_model.fit(X_selected, y, linearregression__sample_weight=weights)

            sel_freq = selection_freq[np.isin(genes_idx, sel_genes)].max(axis=1)
            coef_array = age_model.named_steps['linearregression'].coef_
            print(f'For {fate_name} at age {age}, selected {len(sel_genes)} genes with stability selection frequency above {thresh} and test R^2 score: {age_model.score(X_selected, y, sample_weight=weights):.4f}')

           
            results['age'].append(age)
            results['sel_genes'].append(sel_genes)
            results['freq'].append(sel_freq)
            results['coeffs'].append(coef_array)


        return results


def stability_selection(model, X, y, weights, n_bootstraps=100, sample_fraction=0.5, 
                        random_state=None, model_type='lasso'):

    rng = np.random.RandomState(random_state)
    n_samples = X.shape[0]
    selection_counts = np.zeros(X.shape[1])
    if model_type == 'elasticnetpath':
        # For elastic net path, we will store the coefficients for each alpha along the path
        coef_array = [] 
        alpha_array = []
        frequency_array = []
    else:
        coef_array = np.zeros((n_bootstraps, X.shape[1]))
    for i in range(n_bootstraps):
        bootstrap_idx = rng.choice(n_samples, size=int(sample_fraction * n_samples), replace=False)
        X_bootstrap = X[bootstrap_idx]
        y_bootstrap = y[bootstrap_idx]
        weights_bootstrap = weights[bootstrap_idx]
        if model_type == 'elasticnet':
            model.fit(X_bootstrap, y_bootstrap, sample_weight=weights_bootstrap)
            coef = model.coef_
            coef_array[i, :] = coef
        elif model_type == 'lasso':
            model.fit(X_bootstrap, y_bootstrap, lasso__sample_weight=weights_bootstrap)
            coef = model.named_steps['lasso'].coef_
            coef_array[i, :] = coef
        elif model_type == 'elasticnetpath':
            alphas_enet, coefs_enet = model(X_bootstrap, y_bootstrap, weights_bootstrap)
            coef_array.append(coefs_enet)
            alpha_array.append(alphas_enet)
            frequency_array.append(np.mean(coefs_enet != 0, axis=0))

    if model_type == 'elasticnetpath':
        return alpha_array, coef_array, frequency_array
    


    selection_frequencies =  (coef_array != 0).mean(axis=0)

    return selection_frequencies, coef_array, None



