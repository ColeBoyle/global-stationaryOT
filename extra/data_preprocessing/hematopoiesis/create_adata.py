import pandas as pd
import scipy.io as io
import anndata

print("Loading raw counts...")
counts = io.mmread("GSE189161_matrix.mtx.gz").T.tocsr()
adata = anndata.AnnData(X=counts)

print("Loading metadata...")
adata.obs = pd.read_csv("GSE189161_export_metadata.csv", index_col=0)

print("Loading gene names and Highly Variable Genes...")
genes = pd.read_csv("GSE189161_export_genes.csv", header=None, names=["gene_symbols"])
adata.var = genes
adata.var.index = adata.var['gene_symbols'].values

hvg = pd.read_csv("GSE189161_export_hvg.csv", header=None)[0].values
adata.var['highly_variable'] = adata.var.index.isin(hvg)

print("Loading dimensional reductions...")

adata.obsm['X_pca'] = pd.read_csv("GSE189161_export_pca.csv", index_col=0).values
adata.obsm['X_umap'] = pd.read_csv("GSE189161_export_umap.csv", index_col=0).values

print("Loading integrated data...")

integrated_matrix = io.mmread("GSE189161_export_integrated.mtx").T.tocsr()
adata.obsm['seurat_integrated'] = integrated_matrix

print("AnnData object created successfully:")
print(adata)

adata.write_h5ad('GSE189161_integrated.h5ad')