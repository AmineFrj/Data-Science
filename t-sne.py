# ---------------------------------  Imports ---------------------------

from sklearn.manifold import TSNE
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np
# from ggplot import *

# ----------------- Import Data (from keras datasets) ---------

((X, Y), (testX, testY)) = fashion_mnist.load_data()
X = X.reshape(60000, 784)
# set size to work with small sample
size = 1500 
X,Y = X[:size],Y[:size]

# perform the t_SNE with 2 components (2D)
tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=10000, learning_rate=700)
X_2d = tsne.fit_transform(X)

# to get the total number of labels 
target_ids = np.unique(Y)

 # -------------- plot resulting reduced space -------------
# set size of figure
plt.figure(figsize=(14, 7))
# choose colors (r=red, g=green, .. and so on)
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'cyan', 'orange', 'purple'

for i, c, label in zip(target_ids, colors, target_ids):
    plt.scatter(X_2d[Y == i, 0], X_2d[Y == i, 1], c=c, label=label)
plt.legend()
plt.show()

