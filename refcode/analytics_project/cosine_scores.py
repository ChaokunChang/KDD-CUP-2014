# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import pandas as pd
import os
import re
import time
import argparse
import sys
import re, math
import numpy as np
from collections import Counter

import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')

stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

'''remove punctuation, lowercase, stem'''
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]


def text_to_vector(text):
    if not text:
        return 0
    else:
        words = WORD.findall(text)
        return Counter(words)


if __name__=="__main__":    
    print("Loading files...")
    df1 = pd.read_csv( "essays.csv")
    df2 = pd.read_csv("projects.csv")
    df = pd.merge(df2, df1, how='left', on='projectid') 
    print("Getting previous experience features...")
    df["split"] = "all"
    df["split"][df["date_posted"]<"2010-04-01"] = "none"
    df = df[df["split"]=="all"]
    
    text1 = '';
    text2 = '';
    WORD = re.compile(r'\w+')
    df["score"] = 0;
    for i in range(0,len(df.index)-1):
        text1 = df["short_description"][i];
        text2 = df["essay"][i];
        if text1 is np.nan or text2 is np.nan:
            continue
        
            if text1 != "" or text2 != "":  
                df["score"][i] = cosine_sim(text1, text2) #getting cosine value for two documents 
            
            else:
                continue
    
        
  
 
        
