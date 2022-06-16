# %%
from sklearn import manifold, datasets
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.special import comb

# %%

n_points = 1000
noise = 0.5
X, color = datasets.make_swiss_roll(n_points, noise=noise, random_state=42)


# %%
X.shape

# %%
method = manifold.TSNE(n_components=2,
                       init='random',
                       metric='cosine',
                       random_state=0,
                       learning_rate='auto')


# %%
Y = method.fit_transform(X)

fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(111)
ax.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral);


# %%
d = pdist(X, 'cosine')

# %%
d.shape

# %%
comb(X.shape[0], 2)

# %%
method2 = manifold.TSNE(n_components=2,
                        random_state=0,
                        init='random',
                        learning_rate='auto',
                        metric='precomputed')


# %%
Z = method2.fit_transform(squareform(d))

fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(111)
ax.scatter(Z[:, 0], Z[:, 1], c=color, cmap=plt.cm.Spectral);



