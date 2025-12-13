import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from tqdm import tqdm
from scipy.interpolate import make_splrep
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad

sns.set_style("ticks")
sns.set_palette("deep")

class gene_selection:
    def __init__(self, adata, adata_keys, full_supp=True):

        for key in adata_keys.keys():
            setattr(self, key, adata_keys[key])

        if 'time_key' not in adata_keys.keys():
            print("Must provide time_key in adata_keys")
            return

        self.full_supp = full_supp
        self.adata = adata
        if 'cell_type_key' in adata_keys.keys():
            self.fate_names = np.unique(self.adata.obs[self.cell_type_key])

        metrics = ['sobolev_norm', 'spearmanr_corr']

        self.times = np.unique(self.adata.obs[self.time_key])
        self.results_df = xr.DataArray(
            np.zeros((len(self.adata.var.index), len(self.fate_names), len(metrics))),
            dims=["gene", "fate", "metric"],
            coords={"gene": self.adata.var.index, "fate": self.fate_names, "metric": metrics}
        ) 

    def get_fp_expression_corr(self, label_key=None, fate_names=None, genes='all'):

        if fate_names is None:
            fate_names = self.fate_names

        if label_key is None:
            label_key = self.cell_type_key

        if genes == 'all':
            self.genes_idx = self.adata.var.index
            
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

        self.gene_corrs = []
        for age in tqdm(self.times):
            if self.full_supp:
                adata_a = self.adata[:, self.genes_idx]
                expr = jnp.asarray(adata_a.X.toarray(), dtype=jnp.float32)
                fp = jnp.asarray(adata_a.obsm[f'{label_key}_fp_t={age}'], dtype=jnp.float32)
                weights = jnp.asarray(adata_a.obsp[f'pi_{age}'], dtype=jnp.float32).sum(axis=1)
                # remove nan rows from fp (occurs when fp calculated with HDR_cutoff > 0)
                valid_idx = ~jnp.isnan(fp).any(axis=1)
                expr = expr[valid_idx]
                fp = fp[valid_idx]
                weights = weights[valid_idx].flatten()

                corrs = corr_coef_vmap(expr.T, fp.T, weights)
                # set values outside -1 to 1 range to nan
                corrs = jnp.where((corrs < -1) | (corrs > 1), jnp.nan, corrs)
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

                corrs = corr_coef_vmap(expr.T, fp.T, weights)
                corr_df = pd.DataFrame(corrs, index=adata_a.var.index, columns=fate_names)
                self.gene_corrs.append(corr_df)

        # Format data in xarray
        self.gene_corrs_xr = xr.DataArray(
            np.stack([df.values for df in self.gene_corrs]),
            dims=["age", "gene", "fate"],
            coords={"age": self.times, "gene": self.genes_idx, "fate": fate_names}
        )

    def smooth_gene_corrs(self, k=3, s=None, L2_weighted=False, equal_spacing=True):

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

                sobolev_norm = self.compute_sobolev_norm(B_spline, sample_times[0], sample_times[-1], L2_weighted)
                self.results_df.loc[gene, fate, 'sobolev_norm'] = sobolev_norm 
                spearmanr_corr = pd.Series(y).corr(pd.Series(ft), method='spearman')
                self.results_df.loc[gene, fate, 'spearmanr_corr'] = spearmanr_corr


    def compute_sobolev_norm(self, B_spline, t0, t1, L2_weighted):

        sobolev_norm, _ = quad(lambda x: B_spline.derivative()(x)**2, t0, t1) / (t1 - t0)

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
                    top_genes = driver_genes_a.sort_values(ascending=False).head(n_top_genes).index

                    top_genes_by_fate[fate][age] = pd.DataFrame(index=list(top_genes), columns=['corr'])
                    top_genes_by_fate[fate][age]['corr'] = driver_genes_a.loc[list(top_genes)]


            self.cell_type_top_genes = top_genes_by_fate
 
        else:
            raise ValueError(f"Unknown method: {method}")

        return top_genes_by_fate

    def plot_top_corr_over_age(self, fate_names, n_top_genes=10, cell_types=None, only_TF=False, rank_by_abs_corr=False, smoothed=False):
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
                sobolev_norms = self.results_df.loc[genes, fate, 'sobolev_norm'].values
                marker = 'o'

            fig, ax1 = plt.subplots( figsize=(6, 6))

            for i, gene in enumerate(genes):
                # check if var dataframe has gene_name column
                if self.adata.var.columns.isin(['gene_name']).any():
                    gene_name = self.adata.var.loc[gene]['gene_name']
                else:
                    gene_name = gene
                ax1.plot(t, gene_corr_over_age[i], label=gene_name, marker=marker)

            ax1.set_xlabel('Age')
            ax1.set_ylabel('Correlation b/w gene expression and fate probability')
            if only_TF:
                ax1.set_title(f'Top TF driver genes for {fate}')
            else:
                ax1.set_title(f'Top driver genes for {fate}')
           
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid()
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
            ax1.plot(t, gene_corr_over_age[i], label=gene_name + f' (sobolev norm: {sobolev_norms[i]:.2e},Spearman: {spearmanr_corrs[i]:.2f})', marker=marker)

        ax1.set_xlabel('Age')
        ax1.set_ylabel('Correlation b/w gene expression and fate probability')
        ax1.set_title(f'Correlation trends for {fate}')
    
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid()
        plt.show()
 