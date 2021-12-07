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
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, plot_tree


# In[2]:


def print_scores(y_pred, y_test, print_s=False):
    scores = np.mean(np.array(precision_recall_fscore_support(y_pred, y_test)), axis=1)
    if print_s:
        print(f"F1 Scores:\t{scores[0]}\nPrecision:\t{scores[1]}\nRecall:\t\t{scores[2]}\nSupport:\t{scores[3]}")
    return scores


# In[3]:


train = pd.read_csv('../data/fashion-mnist_train.csv')
test = pd.read_csv('../data/fashion-mnist_test.csv')
y_train = train['label']
X_train = train.loc[:, train.columns != 'label']
X_test, y_test = test.loc[:, test.columns !='label'], test['label']


# In[4]:


print(f"Train set size:{train.shape}")
print(f"Test set size:{test.shape}")
print(f"Training label distribution:{dict(zip(np.unique(y_train, return_counts=True)[0], np.unique(y_train, return_counts=True)[1]))}")
print(f"Testing label distribution:{dict(zip(np.unique(y_test, return_counts=True)[0], np.unique(y_test, return_counts=True)[1]))}")


# In[5]:


clothes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]
d = dict(zip(np.unique(y_test, return_counts=True)[0], np.unique(y_test, return_counts=True)[1]))


# In[6]:


dict(zip(clothes, [1000]*len(clothes)))


# # Logistic Regression

# In[7]:


def run_logistic_regression(depth, X_train, y_train, X_test, y_test):
    # logclf = SGDClassifier(loss="log", max_iter=100)

    clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=depth, multi_class='multinomial')
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    # prediction
    pre = clf.predict(X_test)

    # score
    score = print_scores(pre, y_test)
    if depth == 1 or depth % 50 == 0:
        plot_confusion_matrix(clf, X_test, y_test)
        # print(bias_variance_decomp(LogisticRegression(), X_train=X_train.values, X_test=X_test.values, y_test=y_test.values, y_train=y_train.values, num_rounds=depth))

    return score

d = {"X_train": X_train, "X_test": X_test, "y_train":y_train, "y_test":y_test}
parallel_log = partial(run_logistic_regression, **d)


# In[8]:


with ThreadPoolExecutor(max_workers=20) as tpe:
    out = tpe.map(parallel_log, list(range(0, 100, 10)))

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
        'text': "Logistic Regression Accuracy - MNIST Fashion Classification",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_xaxes(title_text="Max iterations")
fig.update_yaxes(title_text="Score Accuracy")
fig.write_image('mnist_visuals/log_reg_acc.png')
fig.show()


# # Decision Tree

# In[9]:


def run_decision_tree(depth, X_train, y_train, X_test, y_test):
    dtc = DecisionTreeClassifier(max_depth=depth)
    dtc = dtc.fit(X_train, y_train)

    # prediction
    pre = dtc.predict(X_test)

    # score
    score = print_scores(pre, y_test)

    if depth == 1 or depth % 50 == 0:
        plot_confusion_matrix(dtc, X_test, y_test)

    # importance
    importance = [(X_test.columns[e], i) for e, i in enumerate(dtc.feature_importances_) if i > 0]

    return score, importance

d = {"X_train": X_train, "X_test": X_test, "y_train":y_train, "y_test":y_test}
parallel_tree = partial(run_decision_tree, **d)

# run_decision_tree(1, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
# f(2)


# In[10]:


with ThreadPoolExecutor(max_workers=20) as tpe:
    out = tpe.map(parallel_tree, list(range(1, 51)))

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
        'text': "Decision Tree Accuracy - MNIST Fashion Classification",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_xaxes(title_text="Max Tree Depth")
fig.update_yaxes(title_text="Score Accuracy")
fig.write_image('mnist_visuals/dec_tree_acc.png')
fig.show()


# # SVM
# Due to the size and dimensions of the data, a Linear SVM is the only SVM that is suitable to finish in a reasonable amount of time.

# In[37]:


# Linear SVM, no dimension reduction
lin = LinearSVC()
lin.fit(X_train, y_train)
full_pred = lin.predict(X_test)


# In[38]:


print_scores(full_pred, y_test, print_s=True)
plot_confusion_matrix(lin, X_test, y_test)


# In[39]:


pca_X_train = pd.DataFrame(PCA(n_components=1).fit_transform(X_train), columns=['x'])
pca_X_test = pd.DataFrame(PCA(n_components=1).fit_transform(X_test), columns=['x'])


# In[40]:


lin_svc = LinearSVC()
lin_svc.fit(pca_X_train, y_train)


# In[44]:


pred = lin_svc.predict(pca_X_test)
print_scores(pred, y_test, print_s=True)
plot_confusion_matrix(lin_svc, pca_X_test, y_test)


# In[42]:


plot_confusion_matrix(lin_svc, pca_X_test, y_test)


# In[ ]:


pca_X_train = pd.DataFrame(PCA(n_components=78).fit_transform(X_train))
pca_X_test = pd.DataFrame(PCA(n_components=78).fit_transform(X_test))


# In[ ]:


lin_svc = LinearSVC()
lin_svc.fit(pca_X_train, y_train)
pred = lin_svc.predict(pca_X_test)
print_scores(pred, y_test, print_s=True)


# In[ ]:


plot_confusion_matrix(lin_svc, pca_X_test, y_test)


# In[ ]:




