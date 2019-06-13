# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 18:47:15 2016

@author: bbux
"""

import gensim
import pandas as pd
import os
import time
import argparse
import text_util
import sys

"""
Start Script
"""
def main(args):
    if not os.path.exists(args.inputfile):
        raise Exception("Input file does not exist!: " + args.inputfile)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    start = time.time()
    df = pd.read_csv(args.inputfile)
    print("preparing data...")
    text_util.prep_data(df, args.fields)
    print("time: " + str(time.time() - start))
    sys.stdout.flush()
    
    print("Creating text and dictionary for " + str(args.fields) + "...")
    texts, dictionary = text_util.create_texts_and_dictionary(df, args.fields, args.dostem)
    print("time: " + str(time.time() - start))
    sys.stdout.flush()
    
    dict_file_name = args.outdir + "/" + args.prefix + '-dict.txt'
    print("Saving dictionary to " + dict_file_name)
    dictionary.save_as_text(dict_file_name)
    print("time: " + str(time.time() - start))
    sys.stdout.flush()
    
    print("Createing corpus for " + str(args.fields) + "...")
    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]
    print("time: " + str(time.time() - start))
    sys.stdout.flush()
    print("Generating ldamodel for " + str(args.fields) + "...")
    # generate LDA model
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=args.num_topics, 
                                               id2word = dictionary, passes=args.passes)
    print("time: " + str(time.time() - start))
    sys.stdout.flush()
    
    if args.printmodel:
        print(ldamodel.print_topics(num_topics=args.num_topics, num_words=10))

    lda_model_file = args.outdir + "/" + args.prefix + '-lda.model'
    print("Saving ldamodel to " + lda_model_file)
    ldamodel.save(lda_model_file)
    print("Total processing time " + str(time.time() - start) + " for " + str(args.fields))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputfile", help="input csv file", required=True)
    parser.add_argument("-o", "--outdir", help="where to store results", required=True)
    parser.add_argument("-f", "--fields", nargs='+', help="fields to generate models for", required=True)
    parser.add_argument("-p", "--prefix", default="text", help="prefix for naming saved data", required=False)
    parser.add_argument("--printmodel", help="print out the topics", required=False, action="store_true")
    parser.add_argument("--dostem", help="should terms be stemmed, default is no stemming",
                        required=False, action="store_true", default=False)
    parser.add_argument("-n", "--num_topics", help="number of topics to build model with", 
                        required=False, type=int, default=50)
    parser.add_argument("--passes", help="number of passes against data to build model with", 
                        required=False, default=10)
    args = parser.parse_args()
    main(args)
    