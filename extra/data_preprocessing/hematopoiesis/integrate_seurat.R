library(Seurat)
library(Matrix)
library(ggplot2)
sessionInfo()

print("Loading matrix...")
counts_matrix <- readMM("GSE189161_matrix.mtx.gz")

print("Loading cells and features...")
cells <- read.table("GSE189161_cells.txt.gz", header = FALSE, stringsAsFactors = FALSE)
features <- read.table("GSE189161_features.txt.gz", header = FALSE, stringsAsFactors = FALSE)

rownames(counts_matrix) <- features$V1
colnames(counts_matrix) <- cells$V1

print("Loading metadata...")

metadata <- read.csv("GSE189161_metadata.csv.gz", row.names = 1, stringsAsFactors = FALSE)

print("Creating Seurat object...")
seurat_obj <- CreateSeuratObject(
  counts = counts_matrix, 
  meta.data = metadata,
  project = "GSE189161"
)

rm(counts_matrix, cells, features, metadata)
gc()

# Follow procedure in Li et al. 2024
print("Splitting object by sample...")
obj_list <- SplitObject(seurat_obj, split.by = "orig.ident")

# normalize and identify variable features for each sample individually
print("Normalizing and finding variable features...")
obj_list <- lapply(X = obj_list, FUN = function(x) {
  # Normalization to 10k transcripts, log-transformed
  x <- NormalizeData(x, normalization.method = "LogNormalize", scale.factor = 10000)
  # vst method, top 2000 features
  x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = 2000)
  return(x)
})

int_features <- SelectIntegrationFeatures(object.list = obj_list, nfeatures = 2000)

print("Finding integration anchors...")
anchors <- FindIntegrationAnchors(
  object.list = obj_list, 
  anchor.features = int_features, 
  dims = 1:30 # 30 dimensions as specified
)

print("Integrating data...")
seurat_integrated <- IntegrateData(anchorset = anchors, dims = 1:30)

DefaultAssay(seurat_integrated) <- "integrated"

print("Scaling integrated data...")
seurat_integrated <- ScaleData(seurat_integrated)

print("Running PCA...")
seurat_integrated <- RunPCA(seurat_integrated, npcs = 30)


rm(obj_list, anchors, int_features)
gc()

saveRDS(seurat_integrated, file = "GSE189161_integrated.rds")

print("Running UMAP...")
seurat_integrated <- RunUMAP(
  seurat_integrated, 
  reduction = "pca", 
  dims = 1:30 
)


umap_plot <- DimPlot(
  seurat_integrated, 
  reduction = "umap", 
  group.by = "cluster_name", 
  label = TRUE,      
  repel = TRUE   
) + 
  ggtitle("UMAP Integrated Data by Cluster Name")

ggsave(
  filename = "GSE189161_UMAP_cluster_name.png", 
  plot = umap_plot, 
  width = 10, 
  height = 8, 
  dpi = 300
)

saveRDS(seurat_integrated, file = "GSE189161_integrated.rds")

