# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 16:04:34 2014

@author: danielmatthews
"""

import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import metrics

# Read in nflResults.csv
nfl_dat = pd.read_csv('nflResults.csv')


# Data Exploration

plt.scatter(nfl_dat.win, nfl_dat.dn, alpha=0.3)
pd.scatter_matrix(nfl_dat)

# Train Test Split

train, test = train_test_split(nfl_dat,test_size=0.3, random_state=1)

train = pd.DataFrame(data=train, columns=nfl_dat.columns)
test = pd.DataFrame(data=test, columns=nfl_dat.columns)
train
test

feature_cols = ['dn', 'TotYd', 'PassY', 'RushY', 'TO', 'dn_allowed', 'TotYd.1', 'PassY_allowed', 'RushY_allowed', 'TO_def']

X_train = train[feature_cols]
X_test = test[feature_cols]

y_train = train['win']
y_test = test['win']

X = nfl_dat[feature_cols]
y = nfl_dat['win']

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)
bdt.fit(X_train, y_train)
bdt.score(X_test, y_test)

#Adaboost with n_estimators=200
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)
scores = cross_val_score(bdt, X, y, cv=5, scoring='accuracy')
scores
np.mean(scores)

#Adaboost with n_estimators=50
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=50)
scores = cross_val_score(bdt, X, y, cv=5, scoring='accuracy')
scores
np.mean(scores)

#Adaboost with n_estimators=100
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=100)
scores = cross_val_score(bdt, X, y, cv=5, scoring='accuracy')
scores
np.mean(scores)

bdt.fit(X_train, y_train)

#Predictions
y_preds = bdt.predict(X_test)
print metrics.confusion_matrix(y_test, y_preds)

# get predicted probabilities
y_probs = bdt.predict_proba(X_test)[:, 1]
print metrics.roc_auc_score(y_test, y_probs)

# plot ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_probs)
plt.figure()
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

feature_importance = bdt.feature_importances_.tolist()

features = dict(zip(feature_cols, feature_importance))
