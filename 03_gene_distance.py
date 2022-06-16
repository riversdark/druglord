# %%
#default_exp molmap.gene

# %% [markdown]
# # Gene feature processing using MolMap
# 
# > tools for computer aided drug discovery.

# %%
import umap
import numpy as np
import pandas as pd
from lapjv import lapjv
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# %% [markdown]
# data location: `/home/ma/git/druglord/cosine`

# %%
cos00 = np.load('cosine/diag_0_0_cosine_dist.npy')
cos11 = np.load('cosine/diag_1_1_cosine_dist.npy')
cos10 = np.load('cosine/cross_1_0_cosine_dist.npy')

# %%
N, size = cos00.shape

# %%
combined = np.empty((2*size, 2*size))

# %%
combined[:size, :size] = cos00
combined[size:, :size] = cos10
combined[size:, size:] = cos11

# %% [markdown]
# # Dimension reduction using UMAP
# 
# 

# %%
reducer = umap.UMAP(metric='cosine', min_dist=0.1, random_state=1024)


# %% [markdown]
# note that it's important to choose a random state, otherwise the result will vary each time we run the program.

# %%
embedding = reducer.fit_transform(combined)
embedding.shape


# %%
embedding[:5, :5]


# %%
plt.scatter(embedding[:, 0], embedding[:, 1], s=0.05);


# %% [markdown]
# # Grid assignment using LAPJV 
# 
# The cosine distance between two vectors
# 
# $$ \frac{u \cdot v}{{||u||}_2 {||v||}_2}$$

# %%
length = 2 * N

size1 = int(np.ceil(np.sqrt(length)))
size2 = int(np.ceil(length/size1))
grid_size = (size1, size2)

grid = np.dstack(np.meshgrid(np.linspace(0, 1, size2), 
                             np.linspace(0, 1, size1))).reshape(-1, 2)
grid_map = grid[:length]
cost_matrix = cdist(grid_map, embedding, "sqeuclidean").astype(np.float)
cost_matrix = cost_matrix * (100000 / cost_matrix.max())
row_asses, col_asses, _ = lapjv(cost_matrix)



