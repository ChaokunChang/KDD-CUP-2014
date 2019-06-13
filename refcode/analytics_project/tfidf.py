# -*- coding: utf-8 -*-
"""

"""
import re
import os
import sys
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import date

def build_tfidf_models_cv(df, include_val=False, y="y", givens = None, cv=10, 
                          text_vars=["title", "short_description", "need_statement", "essay"]):
    df["r"] = np.random.uniform(0,1,size=len(df))
    if include_val:
        split_tr = (df["split"]=="train") | (df["split"]=="val")
    else:
        split_tr = df["split"]=="train"
    if givens:
        for g in givens:
            split_tr = split_tr & (df[g]==1) 
    y_train = df[y][split_tr].values
    probs = np.arange(0,1. + 1./cv, 1./cv)
    for var in text_vars:
        new_var_name = var+"_pred_partial"
        df[new_var_name] = 0.0
        vectorizer = TfidfVectorizer(min_df=2,
                                     use_idf=1,
                                     smooth_idf=1,
                                     sublinear_tf=1,
                                     ngram_range=(1,2), 
                                     token_pattern=r"(?u)\b[A-Za-z0-9()\'\-?!\"%]+\b",
                                     norm='l2')
        print("attemtping to fit " + var)
        vectorizer.fit(df[var][(df["split"]=="train") | (df["split"]=="val") | (df["split"]=="test")])
        tfidf_train = vectorizer.transform(df[var][split_tr])
        tfidf_all = vectorizer.transform(df[var])        
        lm_model = SGDClassifier(penalty="l2",loss="log",fit_intercept=True, shuffle=True,n_iter=20, n_jobs=-1,alpha=0.000005)
        lm_model.fit(tfidf_train, y_train)
        df[var+"_pred"] = lm_model.predict_proba(tfidf_all)[:,1]
        for i in range(cv):             
            split_train_test = (split_tr) & ((df['r']>=probs[i]) & (df['r']<probs[i+1]))            
            split_train_train = (split_tr) & ((df['r']<probs[i]) | (df['r']>=probs[i+1]))          
            lm_model_temp = SGDClassifier(penalty="l2",loss="log",fit_intercept=True, shuffle=True,n_iter=20, n_jobs=-1,alpha=0.000005)
            x_train_train = vectorizer.transform(df[var][split_train_train])
            x_train_test = vectorizer.transform(df[var][split_train_test])
            lm_model_temp.fit(x_train_train, df[y][split_train_train].values)                       
            pred_train_test = lm_model_temp.predict_proba(x_train_test)[:,1]
            pred_train_train = lm_model_temp.predict_proba(x_train_train)[:,1]
            #df[new_var_name][split_train_test] = pred_train_test                                                
            print('CV: ' + str(i+1))
            print('AUC (Train_Train): ' + str(metrics.roc_auc_score(df['y'][split_train_train],pred_train_train)))
            print('AUC (Train_Test): ' + str(metrics.roc_auc_score(df['y'][split_train_test],pred_train_test)))  
    
def get_length(string):
    return len(string.split())
        
def clean_essay(string, lower=False):
    string = re.sub(r"\\t", " ", string)   
    string = re.sub(r"\\n", " ", string)   
    string = re.sub(r"\\r", " ", string)   
    string = re.sub(r"[^A-Za-z0-9\']", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)
    if lower:
        string = string.lower()
    return string.strip()
    
def bin_date(date_posted):
    if date_posted<"2010-04-01":
        return "none"
    elif date_posted>="2013-01-01":
        return "val"
    elif date_posted>="2014-01-01":
        return "test"
    else:
        return "train"

if __name__=="__main__":    
    folder = sys.argv[1]
    print("Loading files...")
    outcomes_df = pd.read_csv(os.path.join(folder, "outcomes.csv"))
    projects_df = pd.read_csv(os.path.join(folder, "projects.csv"))
    donations_df = pd.read_csv(os.path.join(folder, "donations.csv"))
   
    print("Getting essay features...")
    df = pd.read_csv(os.path.join(folder, "essays.csv"))
    df = pd.merge(df, outcomes_df, how = 'left', on = 'projectid')
    df = pd.merge(df, projects_df, how = 'inner', on = 'projectid')

    # split based on date
    df["split"] = df["date_posted"].apply(bin_date)
    # remove old data
    df = df[df["split"]!="none"]
    
    df["y"] = df["is_exciting"].apply(lambda x: 1 if x == "t" else 0)
    
    text_vars=["title", "short_description", "need_statement", "essay"]
    for var in text_vars:
        df[var][pd.isnull(df[var])] = ""
        df[var] = df[var].apply(clean_essay)
        df[var+"_length"] = df[var].apply(get_length)
        
    build_tfidf_models_cv(df, include_val=False, y="y", 
                          givens=None, cv=10, 
                          text_vars=["title", "short_description", "need_statement", "essay"])
    