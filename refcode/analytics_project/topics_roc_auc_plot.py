# -*- coding: utf-8 -*-
"""

"""
import os
import sys
import pandas as pd

from sklearn import linear_model
from sklearn import cross_validation
from sklearn import metrics

import matplotlib.pyplot as plt
import time


def cross_score(model, predictors, target):
    scores = cross_validation.cross_val_score(model, predictors, 
                                 target, cv=5, scoring='roc_auc')
    log("ROC AUC : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    


def log_metrics(y_test, predictions):
    log(metrics.accuracy_score(y_test, predictions))


def run_model_and_calculate_roc(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    predict = model.predict_proba(x_test)
    
    fpr, tpr, thresholds = metrics.roc_curve(y_test, predict[:,1])
    roc_auc = metrics.auc(fpr, tpr)
        
    return fpr, tpr, roc_auc

def log(message):
    print(message)
    sys.stdout.flush()

folder = sys.argv[1]
topics = sys.argv[2]
start = time.time()
log("Loading files...")

outcomes_df = pd.read_csv(os.path.join(folder, "outcomes.csv"))
topics_df = pd.read_csv(topics)
   
df = pd.merge(outcomes_df, topics_df, how = 'left', on = 'projectid')

df["y"] = df["is_exciting"].apply(lambda x: 1 if x == "t" else 0)
df = df.fillna(0)

target = df.y
cols = [ 'topic' + str(j) for j in range(100)]
predictors = df[cols]

log("Creating LogisticRegression Model...")
logr_model = linear_model.LogisticRegression()

log("Creating SGD Model...")
sdg_model = linear_model.SGDClassifier(penalty="l2",loss="log",fit_intercept=True,
                                       shuffle=True,n_iter=20, n_jobs=-1,alpha=0.000005)

log("Running Logr Model...")
logr_fpr, logr_tpr, logr_roc_auc = run_model_and_calculate_roc(logr_model, predictors, target, predictors, target)

log("Running SGD Model...")
sdg_fpr, sdg_tpr, sdg_roc_auc = run_model_and_calculate_roc(sdg_model, predictors, target, predictors, target)

log("Total processing time " + str(time.time() - start))


plt.title('Receiver Operating Characteristic ')
lineobjs = plt.plot(logr_fpr, logr_tpr, 'b', sdg_fpr, sdg_tpr, 'k')
plt.legend(iter(lineobjs), ('Logr AUC = %0.2f' % logr_roc_auc, 'SDG AUC = %0.2f' % sdg_roc_auc))
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()