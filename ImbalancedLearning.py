##### Load packages ---------------------------------------------------------------------------------------

# Imports from sklearn
from sklearn import svm
from sklearn.dummy import DummyClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

# Imports from other packages
import numpy
import pandas
import imblearn
from matplotlib import pyplot
from collections import Counter
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler





##### Generate data ---------------------------------------------------------------------------------------

# Randomly generate imbalanced data with class and two predictor variables
data_x, data_y = make_classification(n_samples = 10000, n_classes = 2, n_features = 2, n_informative = 2,
                                     n_redundant = 0, n_repeated = 0, weights = [0.99, 0.01], flip_y = 0.005,
                                     random_state = 1, class_sep = 2, n_clusters_per_class = 1)

# Combine data into one array
data = numpy.column_stack((data_x, data_y))

# Split data into training and test set, 50/50 with class (y) stratified
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size = 0.5,
                                                    random_state = 2, stratify = data_y)

# Plot classes: Strong separation but some overlap
fig, ax = pyplot.subplots()
for g in numpy.unique(data_y):
    i = numpy.where(data_y == g)
    ax.scatter(data_x[:, 0][i], data_x[:, 1][i], c = ["blue", "red"][g], label = ["0", "1"][g], alpha = 0.3)
ax.legend()
pyplot.xlabel("X1")
pyplot.ylabel("X2")
pyplot.show()

