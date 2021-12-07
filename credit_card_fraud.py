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
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, plot_confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, plot_tree


# In[2]:


df = pd.read_csv('../data/creditcard.csv')
X = df.loc[:, df.columns != 'Class']
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=4)


# In[11]:


y.value_counts()/len(y)


# ## Undersampling - TODO
# _"We simulate two unbalanced tasks (5% and 25% of positive samples) with overlapping classes and generate a testing set and several training sets from the same distribution."_ - Unbalanced datasets paper
# 
# So create unbalanced tasks

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(18,4))

amount_val = df['Amount'].values
time_val = df['Time'].values

sns.distplot(amount_val, ax=ax[0], color='r')
ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color='b')
ax[1].set_title('Distribution of Transaction Time', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])

plt.show()


# In[4]:


def print_scores(y_pred, y_test, print_s=False):
    scores = np.mean(np.array(precision_recall_fscore_support(y_pred, y_test)), axis=1)
    if print_s:
        print(f"F1 Scores:\t{scores[0]}\nPrecision:\t{scores[1]}\nRecall:\t\t{scores[2]}\nSupport:\t{scores[3]}")
    return scores


# # Logistic Regression

# In[ ]:


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


# In[ ]:


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
        'text': "Logistic Regression Accuracy - Credit Card Fraud Classification",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_xaxes(title_text="Max iterations")
fig.update_yaxes(title_text="Score Accuracy")
fig.write_image('fraud_visuals/log_reg_acc.png')
fig.show()


# In[ ]:


"""eq_1 = X_train.loc[y_train == 1]
eq_0 = X_train.loc[y_train == 0][:len(eq_1)]
even_df = pd.concat([eq_0, eq_1])
even_y = pd.concat([y_train.loc[y_train == 1], y_train.loc[y_train == 0][:len(eq_1)]])
even_logclf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=100, multi_class='multinomial')
even_logclf.fit(even_df, y_train[even_df.index])
even_pred = even_logclf.predict(X_test)
print_scores(even_pred, y_test, print_s=True)
#mapping = dict(zip(np.unique(y), np.unique(even_df.Class)))
even_red = pd.DataFrame(PCA(n_components=2).fit_transform(X_train.sort_index()), columns=['xaxis', 'yaxis']) 
#red['Class'] = list(y_train.sort_index())
#red['Class'] = red['Class'].apply(lambda x: mapping[x])
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=even_red["xaxis"], y=even_y,
                    mode='markers',
                    name='labels'
    )
)
fig.add_trace(
    go.Scatter(x=even_red["xaxis"], y=even_pred,
                    mode='markers',
                    name='predictions',
    )
)     
fig.show()
plot_confusion_matrix(even_logclf, X_test, y_test)"""


# In[ ]:





# # Decision Tree

# In[ ]:


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


# In[ ]:


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
        'text': "Decision Tree Accuracy - Credit Card Fraud Classification",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_xaxes(title_text="Max Tree Depth")
fig.update_yaxes(title_text="Score Accuracy")
fig.write_image('fraud_visuals/dec_tree_acc.png')
fig.show()


# # SVM
# Due to the size and dimensions of the data, a Linear SVM is the only SVM that is suitable to finish in a reasonable amount of time. This helps us avoid the curse of dimensionality.

# In[9]:


lin = LinearSVC()
lin.fit(X_train, y_train)
full_pred = lin.predict(X_test)

print_scores(full_pred, y_test, print_s=True)
plot_confusion_matrix(lin, X_test, y_test)


# In[6]:


pca_X_train = pd.DataFrame(PCA(n_components=1).fit_transform(X_train), columns=['x'])
pca_X_test = pd.DataFrame(PCA(n_components=1).fit_transform(X_test), columns=['x'])


# In[7]:


lin_svc = LinearSVC()
lin_svc.fit(pca_X_train, y_train)
pca_pred = lin_svc.predict(pca_X_test)


# In[8]:


print_scores(pca_pred, y_test, print_s=True)


# In[ ]:


even_svc = LinearSVC()
even_pca_train = pd.DataFrame(PCA(n_components=1).fit_transform(even_df), columns=['x'])
even_svc.fit(even_pca_train, even_y)
peven = even_svc.predict(pca_X_test)


# In[ ]:


print_scores(peven, y_test, print_s=True)


# # Even Prediction accuracy

# In[ ]:


cm = confusion_matrix(np.reshape(peven, (-1, 1)), y_test)


# In[ ]:


plot_confusion_matrix(even_svc, even_pca_train, even_y)

