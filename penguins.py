#!/usr/bin/env python
# coding: utf-8

# In[1]:


from concurrent.futures import ThreadPoolExecutor
from copy import copy
from functools import partial
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import multilabel_confusion_matrix, plot_confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree


# In[2]:


def print_scores(y_pred, y_test, print_s=False):
    scores = np.mean(np.array(precision_recall_fscore_support(y_pred, y_test)), axis=1)
    if print_s:
        print(f"F1 Scores:\t{scores[0]}\nPrecision:\t{scores[1]}\nRecall:\t\t{scores[2]}\nSupport:\t{scores[3]}")
    else:
        return scores


# # ML Algorithms
# 1. Logistic Regression with Stochastic Gradient Descent
# 2. Decision Trees (with and without Adaboost)
# 3. Kernel Support Vector Machines

# # Datasets
# 1. Penguin classification
# 2. MNIST Fashion
# 3. Credit Card Fraud
# 4. Heart Disease

# In[3]:


df = pd.read_csv("/Users/dan/Dropbox/SBU/fall_2021/machine_learning/project/data/penguins_lter.csv")
dft = copy(df)
# dft = dft[['Species', 'Island', 'Region', 'Clutch Completion', 'Culmen Length (mm)', 'Culmen Depth (mm)', "Flipper Length (mm)", "Body Mass (g)", "Sex"]]


# In[4]:


dft['Species'].value_counts()


# In[5]:


def classify_labels(vector: pd.Series):
    if vector.dtype == 'O':
        return vector.astype("category").cat.codes
    else:
        return vector

# Translate strings to integers for matrix crap
dft[dft.columns] = dft[dft.columns].apply(lambda x: classify_labels(x))

# X['Delta 15 N (o/oo)'].dtype and all([np.isnan(i) for i in X['Delta 15 N (o/oo)']])
dft.fillna(value=0,inplace=True)


# show
dft.head()


# In[6]:


# get data
X = dft[dft.columns[~dft.columns.isin(['Species'])]]
y = dft.Species
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)


# In[70]:


np.unique(y_test, return_counts=True)


# # SGD

# In[71]:


"""sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


sgdclf = SGDClassifier(loss="log", max_iter=100)
sgdclf.fit(X_train, y_train)
pred = sgdclf.predict(X_test)
print(f"\n{i}")
print_scores(pred, y_test)"""


# In[72]:


mp = dict(zip(df.Species.astype('category').cat.codes, df.Species))
mp[0]


# # Logistic Regression

# In[73]:


def run_logistic_regression(depth, X_train, y_train, X_test, y_test):
    # logclf = SGDClassifier(loss="log", max_iter=100)

    clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=depth, multi_class='multinomial')
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    # prediction
    pre = clf.predict(X_test)

    # score
    score = print_scores(pre, y_test)
    if depth ==1 or depth % 50 == 0:
        plot_confusion_matrix(clf, X_test, y_test)

    return score

d = {"X_train": X_train, "X_test": X_test, "y_train":y_train, "y_test":y_test}
parallel_log = partial(run_logistic_regression, **d)


# In[74]:


with ThreadPoolExecutor(max_workers=20) as tpe:
    out = tpe.map(parallel_log, list(range(0, 300, 10)))

out_list = [i for i in out]
f1_scores = [i[0] for i in out_list]
precision_scores = [i[1] for i in out_list]
recall_scores = [i[2] for i in out_list]

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=list([10*i for i in range(len(f1_scores))]), y=f1_scores,
                    mode='lines',
                    name='f1 score'
    )
)
fig.add_trace(
    go.Scatter(x=list([10*i for i in range(len(f1_scores))]), y=precision_scores,
                    mode='lines',
                    name='precision'
    )
)
fig.add_trace(
    go.Scatter(x=list([10*i for i in range(len(f1_scores))]), y=recall_scores,
                    mode='lines',
                    name='recall'
    )
)

fig.update_layout(title={
        'text': "Logistic Regression Accuracy - Penguin Classification",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_xaxes(title_text="Max iterations")
fig.update_yaxes(title_text="Score Accuracy")
fig.write_image('penguin_visuals/log_reg_acc.png')
fig.show()


# In[75]:


mapping = dict(zip(np.unique(y), np.unique(df.Species)))
red = pd.DataFrame(PCA(n_components=2).fit_transform(X_train.sort_index()), columns=['xaxis', 'yaxis']) 
red['species'] = list(y_train.sort_index())
red['species'] = red['species'].apply(lambda x: mapping[x])
px.scatter(red, x='xaxis', y='yaxis', color='species', symbol='species')


# In[76]:


fig = go.Figure()
fig.add_trace(
    go.Scatter(x=red["xaxis"], y=red['species'],
                    mode='markers',
                    name='labels'
    )
)
fig.add_trace(
    go.Scatter(x=red["xaxis"], y=[mp[i] for i in pred],
                    mode='markers',
                    name='predictions',
    )
)
fig.show()


# In[77]:


# 3D visualization
red = pd.DataFrame(PCA(n_components=3).fit_transform(X_train), columns=['xaxis', 'yaxis', 'zaxis'])
red['species'] = list(y_train)
red['species'] = red['species'].apply(lambda x: mapping[x])
px.scatter_3d(red, x='xaxis', y='yaxis',z='zaxis', color='species', symbol='species')


# # Decision Tree

# In[78]:


def run_decision_tree(depth, X_train, y_train, X_test, y_test):
    dtc = DecisionTreeClassifier(max_depth=depth)
    dtc = dtc.fit(X_train, y_train)

    # COME BACK TO THIS
    # plot_tree(dtc)

    # prediction
    pre = dtc.predict(X_test)

    # score
    score = print_scores(pre, y_test)

    if depth ==1 or depth % 50 == 0:
        plot_confusion_matrix(dtc, X_test, y_test)

    # importance
    importance = [(X_test.columns[e], i) for e, i in enumerate(dtc.feature_importances_) if i > 0]

    return score, importance

d = {"X_train": X_train, "X_test": X_test, "y_train":y_train, "y_test":y_test}
parallel_tree = partial(run_decision_tree, **d)


# In[79]:


with ThreadPoolExecutor(max_workers=20) as tpe:
    out = tpe.map(parallel_tree, list(range(1, 20)))

out_list = [i for i in out]
f1_scores = [i[0][0] for i in out_list]
precision_scores = [i[0][1] for i in out_list]
recall_scores = [i[0][2] for i in out_list]

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=list(range(len(f1_scores))), y=f1_scores,
                    mode='lines',
                    name='f1 score'
    )
)
fig.add_trace(
    go.Scatter(x=list(range(len(precision_scores))), y=precision_scores,
                    mode='lines',
                    name='precision'
    )
)
fig.add_trace(
    go.Scatter(x=list(range(len(recall_scores))), y=recall_scores,
                    mode='lines',
                    name='recall'
    )
)
fig.update_layout(title={
        'text': "Decision Tree Accuracy - Penguin Classification",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_xaxes(title_text="Max Tree Depth")
fig.update_yaxes(title_text="Score Accuracy")
fig.write_image('penguin_visuals/dec_tree_acc.png')
fig.show()


# Quick conclusion: Very easy to get top performance classifying this data using a light decision tree.
# 
# However, the quick accuracy convergence and size of the dataset indicates to me this algorithm may be subject to high variance error on higher dimensional and larger datasets.
# 
# Additionally, none of the features of the penguins - bill features, weight, height, etc - are much better than flipping a 3-sided coin. Further investigation needed?

# # Kernel SVM

# In[80]:


C = 1.0
svc = SVC(kernel='linear', C=C).fit(X_train, y_train)
pred = svc.predict(X_test)
print(f"Linear Kernel SVM:\n{print_scores(pred, y_test)}")
plot_confusion_matrix(svc, X_test, y_test)


# In[81]:


rbf_svc = SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train, y_train)
pred = rbf_svc.predict(X_test)
print(f"Radial Basis Function SVM:\n{print_scores(pred, y_test)}")
plot_confusion_matrix(rbf_svc, X_test, y_test)


# In[82]:


poly_svc = SVC(kernel='poly', degree=3, C=C).fit(X_train, y_train)
pred = poly_svc.predict(X_test)
print(f"Polynomial Kernel SVM:\n{print_scores(pred, y_test)}")
plot_confusion_matrix(poly_svc, X_test, y_test)

