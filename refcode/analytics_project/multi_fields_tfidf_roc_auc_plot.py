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

from sklearn.feature_extraction.text import TfidfVectorizer

import text_util

def build_tfidf_model(df, var):
    vectorizer = TfidfVectorizer(min_df=2,
                                 use_idf=1,
                                 smooth_idf=1,
                                 sublinear_tf=1,
                                 ngram_range=(1,2), 
                                 token_pattern=r"(?u)\b[A-Za-z0-9()\'\-?!\"%]+\b",
                                 norm='l2')
    log("attemtping to fit " + var)
    vectorizer.fit(df[var])
    return vectorizer.transform(df[var])    


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

log("Loading files...")
outcomes_df = pd.read_csv(os.path.join(folder, "outcomes.csv"))
essay_df = pd.read_csv(os.path.join(folder, "essays.csv"))

df = pd.merge(outcomes_df, essay_df, how = 'left', on = 'projectid')

df["y"] = df["is_exciting"].apply(lambda x: 1 if x == "t" else 0)

log("cleaning text...")
for var in [ "essay", "need_statement", "short_description", "title" ]:
    df[var] = df[var].apply(text_util.clean_essay)
    
logr_results = []
sdg_results = []

for var in [ "essay", "need_statement", "short_description", "title" ]:
#for var in [ "title", "title", "title", "title" ]:
    start = time.time()
    log("building tf-idf vectors for " + var + "...")
    tfidf_vectors = build_tfidf_model(df, var);
    
    target = df.y
    predictors = tfidf_vectors
    
    log("Creating LogisticRegression Model...")
    logr_model = linear_model.LogisticRegression()
    
    log("Creating SGD Model...")
    sdg_model = linear_model.SGDClassifier(penalty="l2",loss="log",fit_intercept=True,
                                           shuffle=True,n_iter=20, n_jobs=-1,alpha=0.000005)
    
    log("Running Logr Model...")
    logr_fpr, logr_tpr, logr_roc_auc = run_model_and_calculate_roc(logr_model, predictors, target, predictors, target)
    logr_results.append((logr_fpr, logr_tpr, logr_roc_auc))
    
    log("Running SGD Model...")
    sdg_fpr, sdg_tpr, sdg_roc_auc = run_model_and_calculate_roc(sdg_model, predictors, target, predictors, target)
    sdg_results.append((sdg_fpr, sdg_tpr, sdg_roc_auc))
    
    log("Total processing time " + str(time.time() - start) + " for " + var)
    

def plot_result(results, model_name):
    plt.title('Receiver Operating Characteristic TF-IDF\n' + model_name)
    lineobjs = plt.plot(results[0][0], results[0][1], 'b',
                        results[1][0], results[1][1], 'k',
                        results[2][0], results[2][1], 'r',
                        results[3][0], results[3][1], 'g')
    labels = ('essay %0.2f' % results[0][2], 
              'need  %0.2f' % results[1][2], 
              'short %0.2f' % results[2][2], 
              'title %0.2f' % results[3][2])
    plt.legend(iter(lineobjs), labels, loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.0])
    plt.ylim([-0.1,1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
plot_result(logr_results, 'Logistic Regression')
plot_result(sdg_results, 'Stochastic Gradient Descent')