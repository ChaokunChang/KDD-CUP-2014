# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 18:47:15 2016

@author: bbux
"""

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora
import gensim
import pandas as pd
import os
import re
import argparse
import time
import sys


def log(message):
    print(message)
    sys.stdout.flush()

def clean_essay(string):
    string = re.sub(r"\\t", " ", string)   
    string = re.sub(r"\\n", " ", string)   
    string = re.sub(r"\\r", " ", string)   
    string = re.sub(r"[^A-Za-z0-9\']", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.lower()
    return string.strip()

def prep_data(df, text_vars):
    for var in text_vars:
            df[var][pd.isnull(df[var])] = ""
            df[var] = df[var].apply(clean_essay)

def create_doc_vectors(df, text_vars, dictionary, stem=False):
    pattern = r"(?u)\b[A-Za-z0-9()\'\-?!\"%]+\b"
    tokenizer = RegexpTokenizer(pattern)
    en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()
    
    texts = []
    for index, row in df.iterrows():
        all_tokens = []
        for var in text_vars:
            tokens = tokenizer.tokenize(row[var])
        
            # remove stop words from tokens
            tokens = [i for i in tokens if not i in en_stop]
            
            # stem tokens
            if stem:
                tokens = [p_stemmer.stem(i) for i in tokens]
        
            all_tokens += tokens
        # add tuple id and bag of words
        doc = dictionary.doc2bow(all_tokens)
        texts.append((row[args.idfield], doc))
    
    return texts


"""
Start Script
"""
def main(args):
    if not os.path.exists(args.inputfile):
        raise Exception("Input file does not exist!: " + args.inputfile)

    log("reading in " + args.inputfile + "...")
    
    df = pd.read_csv(args.inputfile)
    log("preparing data...")
    
    prep_data(df, args.fields)
    
    log("Creating text and loading dictionary for " + str(args.fields) + "...")
    
    dictionary = corpora.Dictionary().load_from_text(args.dictfile)
    
    ldamodel = gensim.models.ldamodel.LdaModel.load(args.modelfile)

    log("creating doc vectors...")
    
    doc_vectors = create_doc_vectors(df, args.fields, dictionary, args.dostem)
    log("converting doc vectors...")
    
    convert(doc_vectors, ldamodel, args.outfile, args.idfield)
    
    
def convert(doc_vectors, ldamodel, outfile, idfield="projectid"):
    topic_data = []
    start = time.time()
    for doc_id, doc in doc_vectors:
        doc_topics = ldamodel[doc]
        # is [ doc_id, 0, 0, ......, 0]
        topic_scores = [doc_id] + ([0] * ldamodel.num_topics)
        for i, v in doc_topics:
            topic_scores[i] = v
        topic_data.append(topic_scores)
        if len(topic_data) % 1000 == 0:
            log("time: " + str(time.time() - start) + " count: " + str(len(topic_data)))
            
            
    column_names = [idfield] + ["topic" + str(i) for i in range(ldamodel.num_topics)]
    df = pd.DataFrame(topic_data, columns=column_names)
    log("saving topic data frame to " + outfile)
    df.to_csv(outfile)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputfile", help="input data file", required=True)
    parser.add_argument("-m", "--modelfile", help="input lda model file", required=True)
    parser.add_argument("-d", "--dictfile", help="input dictonary file", required=True)
    parser.add_argument("-o", "--outfile", help="where to store results", required=True)
    parser.add_argument("-f", "--fields", nargs='+', help="fields to generate models for", required=True)
    parser.add_argument("-p", "--prefix", default="text", help="prefix for naming saved data", required=False)
    parser.add_argument("--idfield", default="projectid", help="id field for data", required=False)
    parser.add_argument("--dostem", help="should terms be stemmed, default is no stemming",
                        required=False, action="store_true", default=False)
    args = parser.parse_args()
    main(args)
    