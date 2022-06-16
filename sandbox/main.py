import os
import umap
import numpy as np
import pandas as pd
from lapjv import lapjv
from scipy.spatial.distance import squareform, cdist
from scipy.cluster.hierarchy import fcluster, linkage


if __name__ == '__main__':

	# Data input
	seed = 15
	cluster_channels = 8
	distance_metric = 'correlation'
	dataX = pd.read_pickle('swissprot_propy_scaled.pkl')

	# Pairwise relationship calculation
	x = dataX.values
	x = (x[..., None] + x[:, None, :]) / 2
	x = np.triu(x).reshape(len(dataX), -1)
	dataX = pd.DataFrame(x, index=dataX.index)
	del x
	dataX = dataX.loc[:, (dataX != 0).any(axis=0)]
	print("--Pairwise relationship calculation: Finish!\n")

	# Pairwise distance calculation
	dataX = dataX.replace([np.nan, np.inf, -np.inf], 0)
	distance_matrix = cdist(dataX.T, dataX.T, metric=distance_metric)
	print("--Pairwise distance calculation: Finish!\n")

	# Dimension reduction
	reducer = umap.UMAP(n_neighbors=30, metric='precomputed', random_state=seed, transform_seed=seed)
	embedding = reducer.fit_transform(distance_matrix)
	print("--Dimension reduction: Finish!\n")

	# Save the results
	map_frame = pd.DataFrame(embedding, index=dataX.columns, columns=["umap_f1", "umap_f2"])

	# Channels split (Hierarchical Cluster)
	print("--applying hierarchical clustering ...\n")
	Z = linkage(squareform(distance_matrix, checks=False), 'complete')
	labels = fcluster(Z, cluster_channels, criterion='maxclust')
	map_frame["Subgroup"] = labels

	# Grid Assignment
	N = len(dataX.columns)

	size1 = int(np.ceil(np.sqrt(N)))
	size2 = int(np.ceil(N/size1))
	grid_size = (size1, size2)

	grid = np.dstack(np.meshgrid(np.linspace(0, 1, size2), np.linspace(0, 1, size1))).reshape(-1, 2)
	grid_map = grid[:N]
	cost_matrix = cdist(grid_map, embedding, metric = "sqeuclidean").astype(np.float64)
	cost_matrix = 100000 * (cost_matrix / cost_matrix.max())
	row_asses, _, _ = lapjv(cost_matrix)

	# Feature assignment based on pairwise distance
	map_frame = map_frame.reindex(dataX.columns[row_asses])
	map_frame.to_csv('assignment.csv', na_rep='NA')
	print("--Grid Assignment: Finish!\n")
