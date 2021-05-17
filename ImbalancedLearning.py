##### Load packages ---------------------------------------------------------------------------------------

# Imports from sklearn
from sklearn import metrics
from sklearn import svm
from sklearn.dummy import DummyClassifier
from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Imports from other packages
import numpy
from pandas import DataFrame
from numpy import sqrt
from numpy import mean
from numpy import argmax
from numpy import arange
from matplotlib import pyplot
from matplotlib import colors
from collections import Counter
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler





##### Generate data ---------------------------------------------------------------------------------------

# Randomly generate imbalanced data with 2 classes and two predictor variables
data_x, data_y = make_classification(n_samples = 10000, n_classes = 2, n_features = 2, n_informative = 2,
                                     n_redundant = 0, n_repeated = 0, weights = [0.99, 0.01], flip_y = 0.006,
                                     random_state = 1, class_sep = 2, n_clusters_per_class = 1)

# Combine data into one array
data = numpy.column_stack((data_x, data_y))

# Split data into training and test set, 50/50 with class (y) stratified
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size = 0.5,
                                                    random_state = 2, stratify = data_y)

# Set colour map for scatterplots
cmap = colors.ListedColormap(["blue", "red"])
bounds = [0, 0.5, 1]
norm = colors.BoundaryNorm(bounds, cmap.N)

# Define function to scatterplot data
def plot_scatter(d_x, d_y, fontsize = 10, legend = True, x1lab = True, x2lab = True,
                 loc = "upper left", binClass = True):
    scatter = pyplot.scatter(d_x[:, 0], d_x[:, 1], c = d_y, cmap = cmap, norm = norm, alpha = 0.3, s = 20,
                             marker = ".", edgecolors = "none")
    if x1lab == True:
        pyplot.xlabel("X1", fontsize = 5)
    if x2lab == True:
        pyplot.ylabel("X2", fontsize = 5)
    if binClass == True:
        pyplot.xlim([0, 5])
        pyplot.ylim([-6, 6])
        pyplot.xticks([0, 1, 2, 3, 4, 5], fontsize = 4)
        pyplot.yticks([-6, -3, 0, 3, 6], fontsize = 4)
    else:
        pyplot.xlim([-6, 12])
        pyplot.ylim([-3, 15])
        pyplot.xticks([-6, -3, 0, 3, 6, 9, 12], fontsize = 4)
        pyplot.yticks([-3, 0, 3, 6, 9, 12, 15], fontsize = 4)
    pyplot.tick_params(length = 2, width = 0.5)
    if legend == True:
        lg = pyplot.legend(*scatter.legend_elements(), fontsize = fontsize, loc = loc, edgecolor = "black",
                      markerscale = 0.3, handletextpad = 0.1, handlelength = 1, framealpha = 1, borderaxespad = 0.3)
        lg.get_frame().set_linewidth(0.3)

# Plot training and test data separately
fig = pyplot.figure(figsize = (3, 1), dpi = 800)
ax = pyplot.subplot(1, 2, 1)
plot_scatter(d_x = train_x, d_y = train_y, fontsize = 4)
ax.text(4.85, -5.5, "Training", fontsize = 4, horizontalalignment = "right")
ax = pyplot.subplot(1, 2, 2)
plot_scatter(d_x = test_x, d_y = test_y, x2lab = False, legend = False)
ax.text(4.85, -5.5, "Test", fontsize = 4, horizontalalignment = "right")
pyplot.tight_layout(pad = 0.4, w_pad = 1.2, h_pad = 1.0)
pyplot.savefig("Plot_Pts.jpeg", dpi = 800, facecolor = "white")





##### Fit discriminant analyses ---------------------------------------------------------------------------

# Define functions for plotting contours and decision boundary
def plot_meshpoints(x, y, h = 0.02):
    x_min, x_max = x.min() - 2, x.max() + 2
    y_min, y_max = y.min() - 2, y.max() + 2
    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h), numpy.arange(y_min, y_max, h))
    return xx, yy
def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(numpy.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

# Compare models using single validation set approach
# For now, we are assuming that no class is more "important" than the other

# Fit LDA and QDA on training data
clfLDA = LinearDiscriminantAnalysis()
clfLDA.fit(train_x, train_y)
clfQDA = QuadraticDiscriminantAnalysis()
clfQDA.fit(train_x, train_y)

# Plot LDA and QDA decision boundaries on top of training data
fig = pyplot.figure(figsize = (3, 1), dpi = 800)
ax = pyplot.subplot(1, 2, 1)
xx, yy = plot_meshpoints(train_x[:, 0], train_x[:, 1])
plot_contours(ax, clfLDA, xx, yy, cmap = pyplot.cm.coolwarm, alpha = 0.4)
plot_scatter(d_x = train_x, d_y = train_y, fontsize = 4)
ax.text(4.85, -5.5, "LDA", fontsize = 4, horizontalalignment = "right")
ax = pyplot.subplot(1, 2, 2)
xx, yy = plot_meshpoints(train_x[:, 0], train_x[:, 1])
plot_contours(ax, clfQDA, xx, yy, cmap = pyplot.cm.coolwarm, alpha = 0.4)
plot_scatter(d_x = train_x, d_y = train_y, x2lab = False, legend = False)
ax.text(4.85, -5.5, "QDA", fontsize = 4, horizontalalignment = "right")
pyplot.tight_layout(pad = 0.4, w_pad = 1.2, h_pad = 1.0)
pyplot.savefig("Plot_DA.jpeg", dpi = 800, facecolor = "white")

# Print F-scores; linear kernel is better
print(metrics.f1_score(train_y, clfLDA.predict(train_x), average = "micro"))
print(metrics.f1_score(train_y, clfQDA.predict(train_x), average = "micro"))

# Print AUPRC; linear kernel is better
precision, recall, _ = precision_recall_curve(train_y, clfLDA.predict(train_x))
print(auc(recall, precision))
precision, recall, _ = precision_recall_curve(train_y, clfQDA.predict(train_x))
print(auc(recall, precision))

# Evaluate linear kernel performance on test data
metrics.confusion_matrix(test_y, clfLDA.predict(test_x))
print(metrics.f1_score(test_y, clfLDA.predict(test_x), average = "micro"))
precision, recall, _ = precision_recall_curve(test_y, clfLDA.predict(test_x))
print(auc(recall, precision))
print(metrics.classification_report(test_y, clfLDA.predict(test_x)))
print(metrics.confusion_matrix(test_y, clfLDA.predict(test_x)))

# Compare models using cross-validation approach

# Create function to run cross-validation and output various statistics
# Split data into training and test; fit model to training data, calculate stats on test data
# Repeat this n_repeats times with 50/50 test/train split
# Accepts model or pipeline objects
def model_cv(obj, n_repeats, metList):
    cv = RepeatedStratifiedKFold(n_splits = 2, n_repeats = n_repeats, random_state = 32463)
    output = cross_validate(obj, data_x, data_y, cv = cv, scoring = metList, n_jobs = -1)
    df = DataFrame(columns = ["metric", "value"])
    for i in metList:
        newdat = DataFrame({"metric":[i], "value":[mean(output["test_" + i])]})
        df = df.append(newdat, ignore_index = True)
    print(df)
    
# Split data into training and test; fit LDA to training, calculate stats on test data
# Repeat this 1000 times; 50/50 test/train split
model_cv(clfLDA, 1000, ["f1_micro", "f1_weighted"])

# Split data into training and test; fit QDA to training, calculate stats on test data
# Repeat this 1000 times; 50/50 test/train split
model_cv(clfQDA, 1000, ["f1_micro", "f1_weighted"])

# Again, we are assuming that no class is more "important" than the other
# Thus, we can compare F1 micro average and find that the linear model performs better





##### Fit support vector machines (cost-insensitive and cost-sensitive) -----------------------------------

# Fit SVM with linear, polynomial (2 and 3), and radial kernels on training data
clfLin = svm.SVC(gamma = "auto", kernel = "linear")
clfLin.fit(train_x, train_y)
clf2dg = svm.SVC(gamma = "auto", kernel = "poly", degree = 2)
clf2dg.fit(train_x, train_y)
clf3dg = svm.SVC(gamma = "auto", kernel = "poly", degree = 3)
clf3dg.fit(train_x, train_y)
clfRbf = svm.SVC(gamma = "auto", kernel = "rbf")
clfRbf.fit(train_x, train_y)

# Same as above, but with approximately 100:1 cost for correctly detecting minority cases
clfLinC = svm.SVC(gamma = "auto", kernel = "linear", class_weight = {0:0.01, 1:0.99})
clfLinC.fit(train_x, train_y)
clf2dgC = svm.SVC(gamma = "auto", kernel = "poly", degree = 2, class_weight = {0:0.01, 1:0.99})
clf2dgC.fit(train_x, train_y)
clf3dgC = svm.SVC(gamma = "auto", kernel = "poly", degree = 3, class_weight = {0:0.01, 1:0.99})
clf3dgC.fit(train_x, train_y)
clfRbfC = svm.SVC(gamma = "auto", kernel = "rbf", class_weight = {0:0.01, 1:0.99})
clfRbfC.fit(train_x, train_y)

# Plot SVM decision boundaries on top of training data
# Left column is cost-insensitive, right is cost-sensitive
fig = pyplot.figure(figsize = (3, 4), dpi = 800)
ax = pyplot.subplot(4, 2, 1)
xx, yy = plot_meshpoints(train_x[:, 0], train_x[:, 1])
plot_contours(ax, clfLin, xx, yy, cmap = pyplot.cm.coolwarm, alpha = 0.4)
plot_scatter(d_x = train_x, d_y = train_y, x1lab = False, fontsize = 4)
ax.text(0.15, -5.5, "Weight 1:1", fontsize = 4, horizontalalignment = "left")
ax.text(4.85, -5.5, "Linear", fontsize = 4, horizontalalignment = "right")
ax = pyplot.subplot(4, 2, 2)
xx, yy = plot_meshpoints(train_x[:, 0], train_x[:, 1])
plot_contours(ax, clfLinC, xx, yy, cmap = pyplot.cm.coolwarm, alpha = 0.4)
plot_scatter(d_x = train_x, d_y = train_y, x1lab = False, x2lab = False, legend = False)
ax.text(0.15, -5.5, "Weight 1:99", fontsize = 4, horizontalalignment = "left")
ax.text(4.85, -5.5, "Linear", fontsize = 4, horizontalalignment = "right")
ax = pyplot.subplot(4, 2, 3)
xx, yy = plot_meshpoints(train_x[:, 0], train_x[:, 1])
plot_contours(ax, clf2dg, xx, yy, cmap = pyplot.cm.coolwarm, alpha = 0.4)
plot_scatter(d_x = train_x, d_y = train_y, x1lab = False, legend = False)
ax.text(0.15, -5.5, "Weight 1:1", fontsize = 4, horizontalalignment = "left")
ax.text(4.85, -5.5, "Poly (2 deg.)", fontsize = 4, horizontalalignment = "right")
ax = pyplot.subplot(4, 2, 4)
xx, yy = plot_meshpoints(train_x[:, 0], train_x[:, 1])
plot_contours(ax, clf2dgC, xx, yy, cmap = pyplot.cm.coolwarm, alpha = 0.4)
plot_scatter(d_x = train_x, d_y = train_y, x1lab = False, x2lab = False, legend = False)
ax.text(0.15, -5.5, "Weight 1:99", fontsize = 4, horizontalalignment = "left")
ax.text(4.85, -5.5, "Poly (2 deg.)", fontsize = 4, horizontalalignment = "right")
ax = pyplot.subplot(4, 2, 5)
xx, yy = plot_meshpoints(train_x[:, 0], train_x[:, 1])
plot_contours(ax, clf3dg, xx, yy, cmap = pyplot.cm.coolwarm, alpha = 0.4)
plot_scatter(d_x = train_x, d_y = train_y, x1lab = False, legend = False)
ax.text(0.15, -5.5, "Weight 1:1", fontsize = 4, horizontalalignment = "left")
ax.text(4.85, -5.5, "Poly (3 deg.)", fontsize = 4, horizontalalignment = "right")
ax = pyplot.subplot(4, 2, 6)
xx, yy = plot_meshpoints(train_x[:, 0], train_x[:, 1])
plot_contours(ax, clf3dgC, xx, yy, cmap = pyplot.cm.coolwarm, alpha = 0.4)
plot_scatter(d_x = train_x, d_y = train_y, x1lab = False, x2lab = False, legend = False)
ax.text(0.15, -5.5, "Weight 1:99", fontsize = 4, horizontalalignment = "left")
ax.text(4.85, -5.5, "Poly (3 deg.)", fontsize = 4, horizontalalignment = "right")
ax = pyplot.subplot(4, 2, 7)
xx, yy = plot_meshpoints(train_x[:, 0], train_x[:, 1])
plot_contours(ax, clfRbf, xx, yy, cmap = pyplot.cm.coolwarm, alpha = 0.4)
plot_scatter(d_x = train_x, d_y = train_y, legend = False)
ax.text(0.15, -5.5, "Weight 1:1", fontsize = 4, horizontalalignment = "left")
ax.text(4.85, -5.5, "Radial (c = 1)", fontsize = 4, horizontalalignment = "right")
ax = pyplot.subplot(4, 2, 8)
xx, yy = plot_meshpoints(train_x[:, 0], train_x[:, 1])
plot_contours(ax, clfRbfC, xx, yy, cmap = pyplot.cm.coolwarm, alpha = 0.4)
plot_scatter(d_x = train_x, d_y = train_y, x2lab = False, legend = False)
ax.text(0.15, -5.5, "Weight 1:99", fontsize = 4, horizontalalignment = "left")
ax.text(4.85, -5.5, "Radial (c = 1)", fontsize = 4, horizontalalignment = "right")
pyplot.tight_layout(pad = 0.4, w_pad = 1.2, h_pad = 1.0)
pyplot.savefig("Plot_SVM.jpeg", dpi = 800, facecolor = "white")

# Compare models using cross-validation approach
# For now, we are assuming that no class is more "important" than the other

# Split data into training and test and fit SVM, then calculate stats on test data
# Repeat this 1000 times with 50/50 test/train split
# Do this for linear kernel, polynomial (2 and 3 degree), and radial kernels
model_cv(clfLin, 1000, ["f1_micro", "f1_weighted"])
model_cv(clf2dg, 1000, ["f1_micro", "f1_weighted"])
model_cv(clf3dg, 1000, ["f1_micro", "f1_weighted"])
model_cv(clfRbf, 1000, ["f1_micro", "f1_weighted"])

# Again, we are assuming that no class is more "important" than the other
# Thus, we can compare F1 micro average and find that the polynomial (2) kernel performs best

# Now, we assume that the "importance" of the minority class is approximately 100:1
# A weighted model would incur a heavy penalty for misclassifying a minority class

# Split data into training and test and fit SVM, then calculate stats on test data
# Repeat this 1000 times with 50/50 test/train split and 100:1 weight on minority class
# Do this for linear kernel, polynomial (2 and 3 degree), and radial kernels
# Reduce reps to 500 since this is more computationally expensive
model_cv(clfLinC, 500, ["balanced_accuracy"])
model_cv(clf2dgC, 500, ["balanced_accuracy"])
model_cv(clf3dgC, 500, ["balanced_accuracy"])
model_cv(clfRbfC, 500, ["balanced_accuracy"])

# Compare balanced accuracy, which inversely weights accuracy based on class size
# The linear kernel performs best





##### ROC and precision-recall curves ---------------------------------------------------------------------

# We need to work with fitted probabilities and thresholds
# Thus, we will use a logistic regression instead of SVM
clfLrg = LogisticRegression(solver = "lbfgs")
clfLrg.fit(train_x, train_y)

# Create no-skill model to compare model performance
clfNull = DummyClassifier(strategy = "stratified")
clfNull.fit(train_x, train_y)
y_hat = clfNull.predict_proba(test_x)
p_null = y_hat[:, 1]

# Find LDA AUC and AUPRC on test data
# Note: AUC is not paritcularly useful here
y_hat = clfLrg.predict_proba(test_x)
p_model = y_hat[:, 1]
print(roc_auc_score(test_y, p_model))
precision, recall, _ = precision_recall_curve(test_y, p_model)
print(auc(recall, precision))

# Plot ROC curve and PR curve, along with no-skill line on test data
# Place point at optimal location based on geometric mean and print optimal threshold
# Note: the ROC curve is not particularly helpful since data is severely imbalanced
fig = pyplot.figure(figsize = (3, 1), dpi = 800)
ax = pyplot.subplot(1, 2, 1)
f_pos, t_pos, _ = roc_curve(test_y, p_null)
pyplot.plot(f_pos, t_pos, linestyle = "-", linewidth = 0.6, label = "No Skill", color = "black")
f_pos, t_pos, thresh = roc_curve(test_y, p_model)
pyplot.plot(f_pos, t_pos, linestyle = "-", linewidth = 0.6, label = "Logistic Regression", color = "green")
pyplot.xticks(fontsize = 4)
pyplot.yticks(fontsize = 4)
pyplot.tick_params(length = 2, width = 0.5)
max_val = argmax(sqrt(t_pos*(1 - f_pos)))
print(thresh[max_val])
pyplot.scatter(f_pos[max_val], t_pos[max_val], marker = ".", color = "black", s = 2)
pyplot.xlabel("1 - Specificity", fontsize = 5)
pyplot.ylabel("Sensitivity", fontsize = 5)
ax = pyplot.subplot(1, 2, 2)
no_skill = len(test_y[test_y == 1])/len(test_y)
pyplot.plot([0, 1], [no_skill, no_skill], linestyle = "-", linewidth = 0.6, label = "No Skill", color = "black")
precision, recall, thresh = precision_recall_curve(test_y, p_model)
pyplot.plot(recall, precision, linestyle = "-", linewidth = 0.6, label = "Logistic Regression", color = "green")
pyplot.xticks(fontsize = 4)
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
pyplot.yticks(fontsize = 4)
pyplot.tick_params(length = 2, width = 0.5)
max_val = argmax((2 * precision * recall)/(precision + recall))
print(thresh[max_val])
pyplot.scatter(recall[max_val], precision[max_val], marker = ".", color = "black", s = 2)
pyplot.xlabel("Recall (Sensitivity)", fontsize = 5)
pyplot.ylabel("Precision (PPV)", fontsize = 5)
lg = pyplot.legend(bbox_to_anchor = (0.14, 0.73, 0.24, 0.2), fontsize = 2)
lg.get_frame().set_linewidth(0.3)
pyplot.tight_layout(pad = 0.4, w_pad = 1.2, h_pad = 1.0)
pyplot.savefig("Plot_Crv.jpeg", dpi = 800, facecolor = "white")

# Note: the data is pretty well-separated, so posterior probabilities are close to 0 or 1
# Thus, moving the threshold will not change much unless close to 0 or 1
# But where is the optimal threshold?

# Define function to convert posterior probabilities to class labels
def prob_to_class(probs, thresh):
	return (probs >= thresh).astype("int")

# Predict class labels and get F-score for default threshold (0.5)
y_hat = clfLrg.predict(test_x)
print(f1_score(test_y, y_hat))

# Predict posterior probabilities
# For each threshold, assign class to probabilities and compute F-score
# Find maximum F-score and threshold at which it occurs
y_hat = clfLrg.predict_proba(test_x)
p_model = y_hat[:, 1]
thresh = arange(0, 1, 0.001)
f_scores = [f1_score(test_y, prob_to_class(p_model, i)) for i in thresh]
print(thresh[argmax(f_scores)])
print(max(f_scores))

# Plot precision and recall versus probability threshold
precision, recall, thresh = precision_recall_curve(test_y, p_model)
pyplot.plot(numpy.append(thresh, 1), precision, linestyle = "-", label = "Precision", color = "green")
pyplot.plot(numpy.append(thresh, 1), recall, linestyle = "-", label = "Recall", color = "blue")
pyplot.xlabel("Threshold")
pyplot.ylabel("Value")
pyplot.legend(bbox_to_anchor = (0.87, 0.75, 0.1, 0.2))
pyplot.show()
pyplot.show()





##### Rebalance data with over-sampling -------------------------------------------------------------------

# Fit to training data and evaluate performance on test data (no rebalancing)
model_cv(clfLrg, 1000, ["f1_micro", "f1_macro", "f1_weighted", "balanced_accuracy"])

# Oversample minority class at 1:1 ratio
# This block of code does not influence models, and just shows how over-sampling works
os = RandomOverSampler(sampling_strategy = "minority")
os_x, os_y = os.fit_resample(train_x, train_y)
print(Counter(train_y))
print(Counter(os_y))

# Fit to training data and evaluate performance on test data (1:1 rebalancing)
pipeline = Pipeline([("samp", RandomOverSampler(sampling_strategy = "minority")),
                     ("model", LogisticRegression(solver = "lbfgs"))])
model_cv(pipeline, 1000, ["f1_micro", "f1_macro", "f1_weighted", "balanced_accuracy"])

# Oversample minority class at 1:4 ratio
# This block of code does not influence models, and just shows how over-sampling works
os = RandomOverSampler(sampling_strategy = 0.25)
os_x, os_y = os.fit_resample(train_x, train_y)
print(Counter(train_y))
print(Counter(os_y))

# Fit to training data and evaluate performance on test data (4:1 rebalancing)
pipeline = Pipeline([("samp", RandomOverSampler(sampling_strategy = 0.25)),
                    ("model", LogisticRegression(solver = "lbfgs"))])
model_cv(pipeline, 1000, ["f1_micro", "f1_macro", "f1_weighted", "balanced_accuracy"])





##### Rebalance data with under-sampling ------------------------------------------------------------------

# Undersample majority class at 1:1 ratio
# This block of code does not influence models, and just shows how under-sampling works
us = RandomUnderSampler(sampling_strategy = "majority")
us_x, us_y = us.fit_resample(train_x, train_y)
print(Counter(train_y))
print(Counter(us_y))

# Fit to training data and evaluate performance on test data (1:1 rebalancing)
pipeline = Pipeline([("samp", RandomUnderSampler(sampling_strategy = "majority")),
                     ("model", LogisticRegression(solver = "lbfgs"))])
model_cv(pipeline, 1000, ["f1_micro", "f1_macro", "f1_weighted", "balanced_accuracy"])

# Unversample majority class at 1:4 ratio
# This block of code does not influence models, and just shows how under-sampling works
us = RandomUnderSampler(sampling_strategy = 0.25)
us_x, us_y = us.fit_resample(train_x, train_y)
print(Counter(train_y))
print(Counter(us_y))

# Fit to training data and evaluate performance on test data (4:1 rebalancing)
pipeline = Pipeline([("samp", RandomUnderSampler(sampling_strategy = 0.25)),
                    ("model", LogisticRegression(solver = "lbfgs"))])
model_cv(pipeline, 1000, ["f1_micro", "f1_macro", "f1_weighted", "balanced_accuracy"])





##### Rebalance data with over- and under-sampling --------------------------------------------------------

# Oversample minority class at 1:4 ratio, then undersample majority class at 1:2 ratio
# This block of code does not influence models, and just shows how under-sampling works
os = RandomOverSampler(sampling_strategy = 0.25)
os_x, os_y = os.fit_resample(train_x, train_y)
us = RandomUnderSampler(sampling_strategy = 0.5)
ous_x, ous_y = us.fit_resample(os_x, os_y)
print(Counter(train_y))
print(Counter(os_y))
print(Counter(ous_y))

# Fit to training data and evaluate performance on test data (4:1 oversampling, 2:1 undersampling)
# Find F-score, precision, and recall
pipeline = Pipeline([("samp1", RandomOverSampler(sampling_strategy = 0.25)),
                     ("samp2", RandomUnderSampler(sampling_strategy = 0.5)),
                     ("model", LogisticRegression(solver = "lbfgs"))])
model_cv(pipeline, 1000, ["f1_micro", "f1_macro", "f1_weighted", "balanced_accuracy"])





##### Multi-class models with imbalanced data -------------------------------------------------------------

# Randomly generate imbalanced data with 4 classes and two predictor variables
data_x, data_y = make_blobs(n_samples = [3500, 3500, 3400, 100], n_features = 2, cluster_std = [1.5, 1.5, 1, 0.8],
                            centers = numpy.array([[1.1, 2.0], [1.5, 8.5], [6.4, 7.4], [5.2, 3.5]]), random_state = 1)

# Combine data into one array
data = numpy.column_stack((data_x, data_y))

# Split data into training and test set, 50/50 with class (y) stratified
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size = 0.5,
                                                    random_state = 2, stratify = data_y)

# Set colour map for scatterplots
cmap = colors.ListedColormap(["blue", "purple", "black", "red"])
bounds = [0, 0.5, 1.5, 2.5, 3]
norm = colors.BoundaryNorm(bounds, cmap.N)

fig = pyplot.figure(figsize = (3, 1), dpi = 800)
ax = pyplot.subplot(1, 2, 1)
plot_scatter(d_x = train_x, d_y = train_y, fontsize = 4, binClass = False)
ax.text(11.7, -2.3, "Training", fontsize = 4, horizontalalignment = "right")
ax = pyplot.subplot(1, 2, 2)
plot_scatter(d_x = test_x, d_y = test_y, binClass = False, x2lab = False, legend = False)
ax.text(11.7, -2.3, "Test", fontsize = 4, horizontalalignment = "right")
pyplot.tight_layout(pad = 0.4, w_pad = 1.2, h_pad = 1.0)
pyplot.savefig("Plot_Pts2.jpeg", dpi = 800, facecolor = "white")

# Fit SVM with varying cost parameter c on training data
clfRC1 = svm.SVC(gamma = "auto", kernel = "rbf", C = 1)
clfRC1.fit(train_x, train_y)
clfRC2 = svm.SVC(gamma = "auto", kernel = "rbf", C = 2)
clfRC2.fit(train_x, train_y)
clfRC3 = svm.SVC(gamma = "auto", kernel = "rbf", C = 5)
clfRC3.fit(train_x, train_y)
clfRC4 = svm.SVC(gamma = "auto", kernel = "rbf", C = 10)
clfRC4.fit(train_x, train_y)

# Same as above, but with approximately 100:1 cost for correctly detecting minority cases
clfRC1C = svm.SVC(gamma = "auto", kernel = "rbf", C = 1, class_weight = {0:0.01, 1:0.01, 2:0.01, 3:0.97})
clfRC1C.fit(train_x, train_y)
clfRC2C = svm.SVC(gamma = "auto", kernel = "rbf", C = 2, class_weight = {0:0.01, 1:0.01, 2:0.01, 3:0.97})
clfRC2C.fit(train_x, train_y)
clfRC3C = svm.SVC(gamma = "auto", kernel = "rbf", C = 5, class_weight = {0:0.01, 1:0.01, 2:0.01, 3:0.97})
clfRC3C.fit(train_x, train_y)
clfRC4C = svm.SVC(gamma = "auto", kernel = "rbf", C = 10, class_weight = {0:0.01, 1:0.01, 2:0.01, 3:0.97})
clfRC4C.fit(train_x, train_y)

# Plot SVM decision boundaries on top of training data
# Left column is cost-insensitive, right is cost-sensitive
fig = pyplot.figure(figsize = (3, 4), dpi = 800)
ax = pyplot.subplot(4, 2, 1)
xx, yy = plot_meshpoints(train_x[:, 0], train_x[:, 1])
plot_contours(ax, clfRC1, xx, yy, cmap = cmap, norm = norm, alpha = 0.4)
plot_scatter(d_x = train_x, d_y = train_y, binClass = False, x1lab = False, fontsize = 4)
ax.text(-5.7, -2.5, "Weight 1:1:1:1", fontsize = 4, horizontalalignment = "left")
ax.text(11.7, -2.5, "Radial (c = 1)", fontsize = 4, horizontalalignment = "right")
ax = pyplot.subplot(4, 2, 2)
xx, yy = plot_meshpoints(train_x[:, 0], train_x[:, 1])
plot_contours(ax, clfRC1C, xx, yy, cmap = cmap, norm = norm, alpha = 0.4)
plot_scatter(d_x = train_x, d_y = train_y, binClass = False, x1lab = False, x2lab = False, legend = False)
ax.text(-5.7, -2.5, "Weight 1:1:1:97", fontsize = 4, horizontalalignment = "left")
ax.text(11.7, -2.5, "Radial (c = 1)", fontsize = 4, horizontalalignment = "right")
ax = pyplot.subplot(4, 2, 3)
xx, yy = plot_meshpoints(train_x[:, 0], train_x[:, 1])
plot_contours(ax, clfRC2, xx, yy, cmap = cmap, norm = norm, alpha = 0.4)
plot_scatter(d_x = train_x, d_y = train_y, binClass = False, x1lab = False, legend = False)
ax.text(-5.7, -2.5, "Weight 1:1:1:1", fontsize = 4, horizontalalignment = "left")
ax.text(11.7, -2.5, "Radial (c = 2)", fontsize = 4, horizontalalignment = "right")
ax = pyplot.subplot(4, 2, 4)
xx, yy = plot_meshpoints(train_x[:, 0], train_x[:, 1])
plot_contours(ax, clfRC2C, xx, yy, cmap = cmap, norm = norm, alpha = 0.4)
plot_scatter(d_x = train_x, d_y = train_y, binClass = False, x1lab = False, x2lab = False, legend = False)
ax.text(-5.7, -2.5, "Weight 1:1:1:97", fontsize = 4, horizontalalignment = "left")
ax.text(11.7, -2.5, "Radial (c = 2)", fontsize = 4, horizontalalignment = "right")
ax = pyplot.subplot(4, 2, 5)
xx, yy = plot_meshpoints(train_x[:, 0], train_x[:, 1])
plot_contours(ax, clfRC3, xx, yy, cmap = cmap, norm = norm, alpha = 0.4)
plot_scatter(d_x = train_x, d_y = train_y, binClass = False, x1lab = False, legend = False)
ax.text(-5.7, -2.5, "Weight 1:1:1:1", fontsize = 4, horizontalalignment = "left")
ax.text(11.7, -2.5, "Radial (c = 5)", fontsize = 4, horizontalalignment = "right")
ax = pyplot.subplot(4, 2, 6)
xx, yy = plot_meshpoints(train_x[:, 0], train_x[:, 1])
plot_contours(ax, clfRC3C, xx, yy, cmap = cmap, norm = norm, alpha = 0.4)
plot_scatter(d_x = train_x, d_y = train_y, binClass = False, x1lab = False, x2lab = False, legend = False)
ax.text(-5.7, -2.5, "Weight 1:1:1:97", fontsize = 4, horizontalalignment = "left")
ax.text(11.7, -2.5, "Radial (c = 5)", fontsize = 4, horizontalalignment = "right")
ax = pyplot.subplot(4, 2, 7)
xx, yy = plot_meshpoints(train_x[:, 0], train_x[:, 1])
plot_contours(ax, clfRC4, xx, yy, cmap = cmap, norm = norm, alpha = 0.4)
plot_scatter(d_x = train_x, d_y = train_y, binClass = False, legend = False)
ax.text(-5.7, -2.5, "Weight 1:1:1:1", fontsize = 4, horizontalalignment = "left")
ax.text(11.7, -2.5, "Radial (c = 10)", fontsize = 4, horizontalalignment = "right")
ax = pyplot.subplot(4, 2, 8)
xx, yy = plot_meshpoints(train_x[:, 0], train_x[:, 1])
plot_contours(ax, clfRC4C, xx, yy, cmap = cmap, norm = norm, alpha = 0.4)
plot_scatter(d_x = train_x, d_y = train_y, binClass = False, x2lab = False, legend = False)
ax.text(-5.7, -2.5, "Weight 1:1:1:97", fontsize = 4, horizontalalignment = "left")
ax.text(11.7, -2.5, "Radial (c = 10)", fontsize = 4, horizontalalignment = "right")
pyplot.tight_layout(pad = 0.4, w_pad = 1.2, h_pad = 1.0)
pyplot.savefig("Plot_SVM2.jpeg", dpi = 800, facecolor = "white")

