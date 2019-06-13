# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 18:47:15 2016

@author: bbux
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


def clean_essay(string):
    string = re.sub(r"\\t", " ", string)   
    string = re.sub(r"\\n", " ", string)   
    string = re.sub(r"\\r", " ", string)   
    string = re.sub(r"[^A-Za-z0-9\']", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)
    string = string.lower()
    return string.strip()

def prep_data(df, text_vars = ["title", "short_description", "need_statement", "essay"]):
    for var in text_vars:
            df[var][pd.isnull(df[var])] = ""
            df[var] = df[var].apply(clean_essay)

def create_dictionary(df, var, stem=False):
    texts = []
    for raw in df[var]:
        tokens = tokenizer.tokenize(raw)
    
        # remove stop words from tokens
        tokens = [i for i in tokens if not i in en_stop]
        
        # stem tokens
        if stem:
            tokens = [p_stemmer.stem(i) for i in tokens]
        
        # add tokens to list
        texts.append(tokens)
    
    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.05, keep_n=100000)
    dictionary.compactify()
    return (texts, dictionary)


"""
Start Script
"""
pattern = r"(?u)\b[A-Za-z0-9()\'\-?!\"%]+\b"
tokenizer = RegexpTokenizer(pattern)

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
    
folder = 'C:/Users/bbux/PersonalDocs/Education/Analytics/ProjectData/sample'
#folder = 'C:/Users/bbux/PersonalDocs/Education/Analytics/ProjectData/alldata'
#folder = 'C:/Users/bbux/PersonalDocs/Education/Analytics/ProjectData/largesample'

df = pd.read_csv(os.path.join(folder, "essays.csv"))
print("preparing data...")
prep_data(df)

text_vars = ["title", "short_description", "need_statement", "essay"]
#text_vars = [ "essay" ]
for var in text_vars:
    start = time.time()
    print("Creating text and dictionary for " + var + "...")
    texts, dictionary = create_dictionary(df, var)
    dictionary.save_as_text(var + '-dict.txt')
    print("Createing corpus for " + var + "...")
    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]
    print("Generating ldamodel for " + var + "...")
    # generate LDA model
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word = dictionary, passes=20)
    print(ldamodel.print_topics(num_topics=10, num_words=10))
    for text in texts[1:3]:
        print(ldamodel[dictionary.doc2bow(text)])
    print("Total processing time " + str(time.time() - start))