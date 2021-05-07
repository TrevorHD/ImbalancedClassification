##### Load packages ---------------------------------------------------------------------------------------

# Imports from sklearn
from sklearn import metrics
from sklearn import svm
from sklearn.dummy import DummyClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

# Imports from other packages
import numpy
import pandas
import imblearn
from numpy import sqrt
from numpy import argmax
from numpy import arange
from matplotlib import pyplot
from collections import Counter
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler





##### Generate data ---------------------------------------------------------------------------------------

# Randomly generate imbalanced data with class and two predictor variables
data_x, data_y = make_classification(n_samples = 10000, n_classes = 2, n_features = 2, n_informative = 2,
                                     n_redundant = 0, n_repeated = 0, weights = [0.99, 0.01], flip_y = 0.006,
                                     random_state = 1, class_sep = 2, n_clusters_per_class = 1)

# Combine data into one array
data = numpy.column_stack((data_x, data_y))

# Split data into training and test set, 50/50 with class (y) stratified
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size = 0.5,
                                                    random_state = 2, stratify = data_y)

# Plot full data
fig, ax = pyplot.subplots()
for g in numpy.unique(data_y):
    i = numpy.where(data_y == g)
    ax.scatter(data_x[:, 0][i], data_x[:, 1][i], c = ["blue", "red"][g], label = ["0", "1"][g], alpha = 0.3)
pyplot.xlabel("X1")
pyplot.ylabel("X2")
pyplot.xlim([0, 5])
pyplot.ylim([-6, 6])
ax.legend()
pyplot.show()

# Plot training data
fig, ax = pyplot.subplots()
for g in numpy.unique(train_y):
    i = numpy.where(train_y == g)
    ax.scatter(train_x[:, 0][i], train_x[:, 1][i], c = ["blue", "red"][g], label = ["0", "1"][g], alpha = 0.3)
pyplot.xlabel("X1")
pyplot.ylabel("X2")
pyplot.xlim([0, 5])
pyplot.ylim([-6, 6])
ax.legend()
pyplot.show()

# Plot test data
fig, ax = pyplot.subplots()
for g in numpy.unique(test_y):
    i = numpy.where(test_y == g)
    ax.scatter(test_x[:, 0][i], test_x[:, 1][i], c = ["blue", "red"][g], label = ["0", "1"][g], alpha = 0.3)
pyplot.xlabel("X1")
pyplot.ylabel("X2")
pyplot.xlim([0, 5])
pyplot.ylim([-6, 6])
ax.legend()
pyplot.show()





##### Fit support vector machines -------------------------------------------------------------------------

# Fit SVM with linear kernel on training data
cflLin = svm.SVC(gamma = "auto", kernel = "linear")
cflLin.fit(train_x, train_y)

# Define functions for plotting contours and decision boundary
def plot_meshpoints(x, y, h=.02):
    x_min, x_max = x.min() - 2, x.max() + 2
    y_min, y_max = y.min() - 2, y.max() + 2
    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h), numpy.arange(y_min, y_max, h))
    return xx, yy
def plot_contours(ax, cflLin, xx, yy, **params):
    Z = cflLin.predict(numpy.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

# Plot linear decision boundary
fig, ax = pyplot.subplots()
X0, X1 = train_x[:, 0], train_x[:, 1]
xx, yy = plot_meshpoints(train_x[:, 0], train_x[:, 1])
plot_contours(ax, cflLin, xx, yy, cmap = pyplot.cm.coolwarm, alpha = 0.4)
for g in numpy.unique(test_y):
    i = numpy.where(test_y == g)
    ax.scatter(test_x[:, 0][i], test_x[:, 1][i], c = ["blue", "red"][g], label = ["0", "1"][g], alpha = 0.3)
ax.set_ylabel("X1")
ax.set_xlabel("X2")
pyplot.xlim([0, 5])
pyplot.ylim([-6, 6])
ax.legend()
pyplot.show()

# Fit SVM with quadratic kernel on training data; plot decision boundary on test data
cfl2dg = svm.NuSVC(nu = 0.009, gamma = "auto", kernel = "poly", degree = 2)
cfl2dg.fit(train_x, train_y)
fig, ax = pyplot.subplots()
X0, X1 = train_x[:, 0], train_x[:, 1]
xx, yy = plot_meshpoints(train_x[:, 0], train_x[:, 1])
plot_contours(ax, cfl2dg, xx, yy, cmap = pyplot.cm.coolwarm, alpha = 0.4)
for g in numpy.unique(test_y):
    i = numpy.where(test_y == g)
    ax.scatter(test_x[:, 0][i], test_x[:, 1][i], c = ["blue", "red"][g], label = ["0", "1"][g], alpha = 0.3)
ax.set_ylabel("X1")
ax.set_xlabel("X2")
pyplot.xlim([0, 5])
pyplot.ylim([-6, 6])
ax.legend()
pyplot.show()

# Fit SVM with cubic kernel on training data; plot decision boundary on test data
cfl3dg = svm.NuSVC(nu = 0.009, gamma = "auto", kernel = "poly", degree = 3)
cfl3dg.fit(train_x, train_y)
fig, ax = pyplot.subplots()
X0, X1 = train_x[:, 0], train_x[:, 1]
xx, yy = plot_meshpoints(train_x[:, 0], train_x[:, 1])
plot_contours(ax, cfl3dg, xx, yy, cmap = pyplot.cm.coolwarm, alpha = 0.4)
for g in numpy.unique(test_y):
    i = numpy.where(test_y == g)
    ax.scatter(test_x[:, 0][i], test_x[:, 1][i], c = ["blue", "red"][g], label = ["0", "1"][g], alpha = 0.3)
ax.set_ylabel("X1")
ax.set_xlabel("X2")
pyplot.xlim([0, 5])
pyplot.ylim([-6, 6])
ax.legend()
pyplot.show()

# Fit SVM with radial kernel on training data; plot decision boundary on test data
cflRbf = svm.NuSVC(nu = 0.009, gamma = "auto", kernel = "rbf")
cflRbf.fit(train_x, train_y)
fig, ax = pyplot.subplots()
X0, X1 = train_x[:, 0], train_x[:, 1]
xx, yy = plot_meshpoints(train_x[:, 0], train_x[:, 1])
plot_contours(ax, cflRbf, xx, yy, cmap = pyplot.cm.coolwarm, alpha = 0.4)
for g in numpy.unique(test_y):
    i = numpy.where(test_y == g)
    ax.scatter(test_x[:, 0][i], test_x[:, 1][i], c = ["blue", "red"][g], label = ["0", "1"][g], alpha = 0.3)
ax.set_ylabel("X1")
ax.set_xlabel("X2")
pyplot.xlim([0, 5])
pyplot.ylim([-6, 6])
ax.legend()
pyplot.show()





##### ROC and precision-recall curves ---------------------------------------------------------------------

# Get classification report on test data
metrics.classification_report(test_y, cflLin.predict(test_x))

# Get confusion matrix on test data
metrics.confusion_matrix(test_y, cflLin.predict(test_x))

# Get overall accuracy on test data
# Note: this is not helpful since data is severely imbalanced
metrics.accuracy_score(test_y, cflLin.predict(test_x))

# Note: we can do the above for just about any model
# If we want to plot ROC and PR curves, we need posterior probabilities
# SVM does not produce these, so we will use a logistic regression

# Create no-skill model to compare model performance
cflNull = DummyClassifier(strategy = "stratified")
cflNull.fit(train_x, train_y)
y_hat = cflNull.predict_proba(test_x)
p_null = y_hat[:, 1]

# Fit logistic regression; find AUC and AUPRC
model = LogisticRegression(solver = "lbfgs")
model.fit(train_x, train_y)
y_hat = model.predict_proba(test_x)
p_model = y_hat[:, 1]
print(roc_auc_score(test_y, p_model))
precision, recall, _ = precision_recall_curve(test_y, p_model)
print(auc(recall, precision))

# Plot ROC curve and no-skill line
# Place point at optimal location based on geometric mean and print optimal threshold
# Note: this is not helpful since data is severely imbalanced
f_pos, t_pos, _ = roc_curve(test_y, p_null)
pyplot.plot(f_pos, t_pos, linestyle = "--", label = "No Skill", color = "black")
f_pos, t_pos, thresh = roc_curve(test_y, p_model)
pyplot.plot(f_pos, t_pos, marker = ".", label = "Logistic Regression", color = "green")
max_val = argmax(sqrt(t_pos*(1 - f_pos)))
print(thresh[max_val])
pyplot.scatter(f_pos[max_val], t_pos[max_val], marker = "o", color = 'black')
pyplot.xlabel("1 - Specificity")
pyplot.ylabel("Sensitivity")
pyplot.legend()
pyplot.show()

# Plot PR curve and no-skill line
# Place point at optimal location based on F-score and print optimal threshold
no_skill = len(test_y[test_y == 1])/len(test_y)
pyplot.plot([0, 1], [no_skill, no_skill], linestyle = "--", label = "No Skill", color = "black")
precision, recall, thresh = precision_recall_curve(test_y, p_model)
pyplot.plot(recall, precision, marker = ".", label = "Logistic Regression", color = "green")
max_val = argmax((2 * precision * recall)/(precision + recall))
print(thresh[max_val])
pyplot.scatter(recall[max_val], precision[max_val], marker = "o", color = 'black')
pyplot.xlabel("Recall (Sensitivity)")
pyplot.ylabel("Precision (PPV)")
pyplot.legend(bbox_to_anchor = (0.03, 0.73, 0.4, 0.2))
pyplot.show()

# Define function to convert posterior probabilities to class labels
def prob_to_class(probs, thresh):
	return (probs >= thresh).astype("int")

# Predict class labels and get F-score for default threshold (0.5)
y_hat = model.predict(test_x)
print(f1_score(test_y, y_hat))

# Predict posterior probabilities
# For each threshold, assign class to probabilities and compute F-score
# Find maximum F-score and threshold at which it occurs
y_hat = model.predict_proba(test_x)
p_model = y_hat[:, 1]
thresh = arange(0, 1, 0.001)
f_scores = [f1_score(test_y, prob_to_class(p_model, i)) for i in thresh]
print(thresh[argmax(f_scores)])
print(f_scores[argmax(f_scores)])
















