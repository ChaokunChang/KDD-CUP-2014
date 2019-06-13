# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 18:47:15 2016

@author: bbux
"""


from gensim import corpora, models
from operator import itemgetter
import gensim
import pandas as pd
import os
import argparse
import text_util

"""
Start Script
"""
def main(args):
    if not os.path.exists(args.inputfile):
        raise Exception("Input file does not exist!: " + args.inputfile)

    df = pd.read_csv(args.inputfile)
    print("preparing data...")
    text_util.prep_data(df, args.fields)
    
    print("Creating text and loading dictionary for " + str(args.fields) + "...")
    dictionary = corpora.Dictionary().load_from_text(args.dictfile)
    
    ldamodel = gensim.models.ldamodel.LdaModel.load(args.modelfile)
    
    texts = text_util.create_texts(df, args.fields, args.dostem)
    for text in texts[0:10]:
        doc_topics = ldamodel[dictionary.doc2bow(text)]
        srtd = sorted(doc_topics, key=itemgetter(1), reverse=1)
        print(text)
        size = 3 if len(srtd) > 3 else len(srtd)
        for topic_id, score in srtd[0:size]:
            print(str(score))
            print(ldamodel.print_topic(topic_id))
        print("--------------------------------")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputfile", help="input data file", required=True)
    parser.add_argument("-m", "--modelfile", help="input lda model file", required=True)
    parser.add_argument("-d", "--dictfile", help="input dictonary file", required=True)
    parser.add_argument("-f", "--fields", nargs='+', help="fields to generate models for", required=True)
    parser.add_argument("--dostem", help="should terms be stemmed, default is no stemming",
                        required=False, action="store_true", default=False)
    args = parser.parse_args()
    main(args)
    
