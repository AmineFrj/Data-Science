# ---------------------------------  Imports ---------------------------

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from keras.datasets import fashion_mnist
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

# ----------------- Import Data (from keras datasets) ---------

((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
# reshape data => from 28X28 to 784 & reduce size
size = 10
trainX, testX = np.split(trainX.reshape(60000, 784), size)[
    0], np.split(testX.reshape(10000, 784), size)[0]
trainY, testY = np.split(trainY, size)[0], np.split(testY, size)[0]

# ------- Get number of distinct classes --------
nc = len(np.unique(trainY))

# ----------------- Perform LDA ------------------
# first parameter n_components (at lest nc-1)
numComponents = nc-1

lda = LDA(n_components=numComponents)
X_train = lda.fit_transform(np.float64(trainX), trainY)
# We need to transfort the test data also 
X_test = lda.transform(np.float64(testX))

# Predicting the Test set results
classifier = RandomForestClassifier()
classifier.fit(X_train, trainY)
y_pred = classifier.predict(X_test)

confMat = confusion_matrix(testY, y_pred)
print(confMat)
print('Accuracy:', str(accuracy_score(testY, y_pred)))

 # -------------- plot resulting clustering - using t-SNE -------------

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
X_2d = tsne.fit_transform(X_test)

# to match the labels of fashion mnist
labels = ['t_shirt_top', 'trouser', 'pullover', 'dress',
          'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']

# set size of figure
plt.figure(figsize=(14, 7))
# choose colors (r=red, g=green, .. and so on)
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'cyan', 'orange', 'purple'
target_ids = np.unique(testY)

for i, c, label in zip(target_ids, colors, labels):
    plt.scatter(X_2d[y_pred == i, 0], X_2d[y_pred == i, 1], c=c, label=label)
plt.legend()
plt.show()
