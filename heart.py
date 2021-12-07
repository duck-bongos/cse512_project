#!/usr/bin/env python
# coding: utf-8

# In[1]:


from concurrent.futures import ThreadPoolExecutor
from copy import copy
from functools import partial
import os
import time

import matplotlib.pyplot as plt
from mlxtend.evaluate import bias_variance_decomp
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import multilabel_confusion_matrix, plot_confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, plot_tree


# In[4]:


def print_scores(y_pred, y_test, print_s=False):
    scores = np.mean(np.array(precision_recall_fscore_support(y_pred, y_test)), axis=1)
    if print_s:
        print(f"F1 Scores:\t{scores[0]}\nPrecision:\t{scores[1]}\nRecall:\t\t{scores[2]}\nSupport:\t{scores[3]}")
    return scores


# In[5]:


def classify_labels(vector: pd.Series):
    if vector.dtype == 'O':
        return vector.astype("category").cat.codes
    else:
        return vector


# In[6]:


# load data
df = pd.read_csv("../data/heart.csv")

# Translate strings to integers for matrix crap
df[df.columns] = df[df.columns].apply(lambda x: classify_labels(x))
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)


# # Logistic Regression

# In[55]:


def run_logistic_regression(depth, X_train, y_train, X_test, y_test):
    # logclf = SGDClassifier(loss="log", max_iter=100)

    clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=depth, multi_class='multinomial')
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    # prediction
    pre = clf.predict(X_test)

    # score
    score = print_scores(pre, y_test)
    if depth % 50 == 0:
        plot_confusion_matrix(clf, X_test, y_test)
        print(bias_variance_decomp(LogisticRegression(), X_train=X_train.values, X_test=X_test.values, y_test=y_test.values, y_train=y_train.values, num_rounds=depth))

    return score

d = {"X_train": X_train, "X_test": X_test, "y_train":y_train, "y_test":y_test}
parallel_log = partial(run_logistic_regression, **d)


# In[56]:


with ThreadPoolExecutor(max_workers=20) as tpe:
    out = tpe.map(parallel_log, list(range(0, 500, 10)))

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
        'text': "Logistic Regression Accuracy - Heart Disease Classification",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_xaxes(title_text="Max iterations")
fig.update_yaxes(title_text="Score Accuracy")
fig.write_image('heart_visuals/log_reg_acc.png')
fig.show()


# # Decision Tree

# In[ ]:


def run_decision_tree(depth, X_train, y_train, X_test, y_test):
    dtc = DecisionTreeClassifier(max_depth=depth)
    dtc = dtc.fit(X_train, y_train)

    # COME BACK TO THIS
    # plot_tree(dtc)

    # prediction
    pred = dtc.predict(X_test)

    # score
    score = print_scores(pred, y_test)

    # importance
    importance = [(X_test.columns[e], i) for e, i in enumerate(dtc.feature_importances_) if i > 0]

    return score, importance

d = {"X_train": X_train, "X_test": X_test, "y_train":y_train, "y_test":y_test}
parallel_tree = partial(run_decision_tree, **d)
# run_decision_tree(1, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
# f(2)


# In[ ]:


with ThreadPoolExecutor(max_workers=20) as tpe:
    out = tpe.map(parallel_tree, list(range(1, 50)))

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
        'text': "Decision Tree Accuracy - Heart Classification",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_xaxes(title_text="Max Tree Depth")
fig.update_yaxes(title_text="Score Accuracy")
fig.write_image('heart_visuals/dec_tree_acc.png')
fig.show()


# In[ ]:


with ThreadPoolExecutor(max_workers=20) as tpe:
    out = tpe.map(parallel_func, list(range(1, 20)))

out_list = [i for i in out]


# In[ ]:


f1_scores = [i[0][0] for i in out_list]
precision_scores = [i[0][1] for i in out_list]
recall_scores = [i[0][2] for i in out_list]


# In[ ]:


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
        'text': "Decision Tree Accuracy - Heart Classification",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_xaxes(title_text="Max Tree Depth")
fig.update_yaxes(title_text="Score Accuracy")
fig.write_image('heart_visuals/dec_tree_acc.png')
fig.show()


# # SVMs

# In[ ]:


C = 1.0
svc = SVC(kernel='linear', C=C).fit(X_train, y_train)
rbf_svc = SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train, y_train)
poly_svc = SVC(kernel='poly', degree=3, C=C).fit(X_train, y_train)
models = (svc, rbf_svc, poly_svc)
names = ("Linear SVM", "Radial Basis Function SVM", "Polynomial SVM")
for m, n in zip(models, names):
    pred = m.predict(X_test)
    print(f"{n}\n{'-'*60}")
    print_scores(y_pred=pred, y_test=y_test, print_s=True)
    print(f"\n{'-'*60}\n")

