# ---------------------------------  Imports ---------------------------

from sklearn import svm, metrics
from keras.datasets import fashion_mnist
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

# ----------------- Import Data (from keras datasets) ---------

((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
# to match the labels of fashion mnist
labels = ['t_shirt_top', 'trouser', 'pullover', 'dress',
          'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
size = 10 # to reduce the size of data
trainX, reshaped_trainX = np.split(trainX.reshape(60000, 784), size)[
    0], np.split(testX.reshape(10000, 784), size)[0]
trainY, testY = np.split(trainY, 10)[0], np.split(testY, size)[0]
n_samples = len(trainX)

# Create a classifier: a support vector classifier
classifier = svm.SVC(C=1.0,gamma='scale', kernel='rbf')

# Let the classifier learn from the training data
classifier.fit(trainX, trainY)

# Now predict the value of test data
predicted = classifier.predict(reshaped_trainX)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(testY, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(testY, predicted))

# Now we plot a sample of predicted data
images_and_predictions = list(zip(testX, predicted))

for index, (image, prediction) in enumerate(images_and_predictions[:8]):
    plt.subplot(2, 4, index+1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(labels[int(prediction)])
plt.ylim("amine")
plt.legend()
plt.show()

 # -------------- plot resulting clustering - using t-SNE -------------

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
X_2d = tsne.fit_transform(reshaped_trainX)

# set size of figure
plt.figure(figsize=(14, 7))
# choose colors (r=red, g=green, .. and so on)
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'cyan', 'orange', 'purple'

for i, c, label in zip(target_ids, colors, labels):
    plt.scatter(X_2d[predicted == i, 0], X_2d[predicted == i, 1], c=c, label=label)
plt.legend()
plt.show()
