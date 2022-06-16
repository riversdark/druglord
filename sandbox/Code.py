import umap
import numpy as np
import pandas as pd
from lapjv import lapjv
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster, linkage

#Data input
file = "CN-AML"
cluster_channels = 7
f_percentage = 0.1 #feature selection: (0,1], total number: 18532

data_T = pd.read_csv("{}_log2expression-response.csv".format(file), header = 0, index_col = 0)

#Feature Matrix and Y_true labels
dataX = data_T.drop(columns = "response")
dataY = data_T["response"]

#Feature variance sorting
VAR_list = np.var(dataX, axis = 0)
f_sorted = np.argsort(VAR_list)[::-1]
f_name = dataX.columns[f_sorted]

f_num = round(f_percentage * len(f_name))
dataX = dataX.reindex(columns = f_name[:f_num])

# Pairwise relationship calculation
x = dataX.values
x = (x[..., None] + x[:, None, :]) / 2
x = np.triu(x).reshape(len(dataX), -1)
dataX = pd.DataFrame(x, index=dataX.index)
del x
dataX = dataX.loc[:, (dataX != 0).any(axis=0)]
print("--Pairwise relationship calculation: Finish!\n")

#Pairwise distance calculation
dataX = dataX.replace([np.nan, np.inf, -np.inf], 0)
distance_matrix = pairwise_distances(dataX.T, metric = 'cosine')
print("--Pairwise distance calculation: Finish!\n")

#Dimension reduction
reducer = umap.UMAP(n_neighbors = 30, min_dist = 0.1, n_components = 2, metric = 'precomputed', random_state = 1)
embedding = reducer.fit_transform(distance_matrix)
print("--Dimension reduction: Finish!\n")

#Save the results
map_frame = pd.DataFrame(embedding, index = dataX.columns, columns = ["umap_f1","umap_f2"])

#Channels split (Hierarchical Cluster)
print("--applying hierarchical clustering ...\n")
Z = linkage(squareform(distance_matrix, checks=False), 'complete')
labels = fcluster(Z, cluster_channels, criterion='maxclust')
map_frame["Subgroup"] = labels

#Grid Assignment
N = len(dataX.columns)

size1 = int(np.ceil(np.sqrt(N)))
size2 = int(np.ceil(N/size1))
grid_size = (size1, size2)

grid = np.dstack(np.meshgrid(np.linspace(0, 1, size2), np.linspace(0, 1, size1))).reshape(-1, 2)
grid_map = grid[:N]
cost_matrix = pairwise_distances(grid_map, embedding, metric = "sqeuclidean").astype(np.float64)
cost_matrix = 100000 * (cost_matrix / cost_matrix.max())
row_asses, col_asses, lap_z = lapjv(cost_matrix)

#Feature assignment based on pairwise distance
map_frame = map_frame.reindex(dataX.columns[row_asses])
map_frame.to_csv('{}_2D_assigned_{}X{}_fmap_with_subgroup.csv'.format(file,size1,size2), na_rep='NA')
print("--Grid Assignment: Finish!\n")
