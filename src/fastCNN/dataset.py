import os
import string
import time
import math
import os.path as op
from os.path import join as opj
import pickle as pkl

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# from numpy.random import RandomState
# from torch.utils.data import Dataset
# from torch.autograd import Variable

import fastNLP
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
from fastNLP import Const


def combine_whitespace(s):
    return s.split()


def tokenize(data):
    print("Tokenizing data num:{}".format(len(data)))
    new_data = []
    for i in range(len(data)):
        text = data[i]
        if text == "":
            print("Empty text.{0}".format(i))
        elif text is None:
            print("None type text.")
        elif not isinstance(text,str):
            print("ATTENTION TYPE:{} of type {}".format(text,type(text)))
            text = str(text)
        newtext = ""
        for c in text:
            if c not in string.punctuation:
                newtext += c
            else:
                newtext += ' '
        newtext = combine_whitespace(newtext.lower())
        if newtext is not None:
            new_data.append(newtext)
    print("Tokenized:{}/{}".format(len(new_data),len(data)))
    return new_data


def text2multi_hot(words, vocab_size, word2index=None):
    multi_hot_vector = [0]*(vocab_size)
    for word in words:
        multi_hot_vector[word] = 1
    return multi_hot_vector


def class2target(class_type, class_num):
    target = [0]*class_num
    target[class_type] = 1
    return target


class TextData():
    data_src = "all_data"
    class_num = 2
    min_count = 10
    max_seq_len = 500
    seq_limit = 2000

    train_set = DataSet()
    val_set = DataSet()
    test_set = DataSet()
    train_size = 0
    val_size = 0
    test_size = 0

    test_projectid = None

    vocab = None
    vocab_size = 0

    def __init__(self, data_src="all_data", min_count=10, seq_limit=None):
        self.data_src = data_src
        self.min_count = min_count
        if seq_limit is not None:
            self.seq_limit = seq_limit

    def find_max_len(self, words):
        self.max_seq_len = max(len(words), self.max_seq_len)

    def seq_regularize(self, words):
        wlen = len(words)
        if wlen < self.max_seq_len:
            return [0]*(self.max_seq_len-wlen) + words
        else:
            return words[:self.max_seq_len]

    def fetch_csv(self, path, subset_num=None,us_rate=None,os_rate=None):
        """ 
        us_rate: under sampling rate
        os_rate: over sampling rate
         """
        print("Loading data from {} .......".format(path))
        df = pd.read_csv(path, header=0)
        # text_vars=["title", "short_description", "need_statement", "essay"]
        text_vars = "essay"  # only select the essay column
        target_var = "y"
        train_df = df[df['split'] == 'train']
        val_df = df[df['split'] == 'val']
        test_df = df[df['split'] == 'test']
        train_num = len(train_df)
        val_num = len(val_df)
        test_num = len(test_df)
        print("nums:({},{},{})".format(train_num,val_num,test_num))
        if os_rate is not None:
            print("Over Sampling...")
            ros = RandomOverSampler(random_state=0)
        elif us_rate is not None:
            print("Under Sampling...")
            train_df_t = train_df[df['is_exciting'] == 't']
            train_df_f = train_df[df['is_exciting'] == 'f']
            t_num = len(train_df_t)
            f_num = len(train_df_f)
            print("Raw train t:f = {}:{}".format(t_num,f_num))
            nf_num = int(t_num / us_rate)
            f_num = min(nf_num,f_num)
            balanced_train_t = train_df_t.sample(n=t_num)
            balanced_train_f = train_df_f.sample(n=f_num)
            train_df = pd.concat([balanced_train_t,balanced_train_f]).sample(frac=1)
            print("Balanced train: t:f = {}:{}".format(len(balanced_train_t),len(balanced_train_f) ))
            # print("Train 1.0:",len(train_df[train_df[target_var] == 1.0]))

            val_df_t = val_df[df['is_exciting'] == 't']
            val_df_f = val_df[df['is_exciting'] == 'f']
            t_num = len(val_df_t)
            f_num = len(val_df_f)
            print("Raw val t:f = {}:{}".format(t_num,f_num))
            nf_num = int(t_num / us_rate)
            f_num = min(nf_num,f_num)
            balanced_val_t = val_df_t.sample(n=t_num)
            balanced_val_f = val_df_f.sample(n=f_num)
            val_df = pd.concat([balanced_val_t,balanced_val_f]).sample(frac=1)
            print("Balanced val: t:f = {}:{}".format(len(balanced_val_t) ,len(balanced_val_f) ))
        else:
            print("No sampling")
        if subset_num is not None and subset_num > 0:
            print("Get sub set of size {}.".format(subset_num))
            train_df = train_df.sample(n=subset_num)
            val_df = val_df.sample(n=subset_num)
        
        train_num = len(train_df)
        val_num = len(val_df)
        test_num = len(test_df) 
        print("nums:({},{},{})".format(train_num,val_num,test_num))

        print("tokenize train set")
        train_input = tokenize(train_df[text_vars].values)
        print("tokenize val set")
        val_input = tokenize(val_df[text_vars].values)
        print("tokenize test set")
        test_input = tokenize(test_df[text_vars].values)

        train_target = train_df[target_var].values
        val_target = val_df[target_var].values
        test_target = test_df[target_var].values
        assert (self.class_num == 2)
        self.test_projectid = test_df['projectid']
        # Building Fastnlp dataset.
        print("Building Fastnlp dataset.")
        if os_rate is not None:
            train_input,train_target = ros.fit_sample(  np.array(train_input)[:,np.newaxis],
                                                        np.array(train_target)[:,np.newaxis])
            train_input = train_input.squeeze().tolist()
            train_target = train_target.tolist()
            val_input,val_target = ros.fit_sample(  np.array(val_input)[:,np.newaxis],
                                                        np.array(val_target)[:,np.newaxis])
            val_input = val_input.squeeze().tolist()
            val_target = val_target.tolist()
        self.train_set = DataSet({"text": train_input, "class": train_target})
        self.val_set = DataSet({"text": val_input, "class": val_target})
        self.test_set = DataSet({"text": test_input, "class": test_target})

        # Building Fastnlp vocabulary...
        print("Building Fastnlp vocabulary.")
        self.vocab = Vocabulary(min_freq=self.min_count)
        self.train_set.apply(
            lambda x: [self.vocab.add_word(word) for word in x['text']])
        self.vocab.build_vocab()
        self.vocab.build_reverse_vocab()
        self.vocab_size = len(self.vocab)
        # Building multi-hot-vector for train_set and test_set.
        print("Building id-presentation for train_set and test_set.")
        self.vocab.index_dataset(self.train_set, self.val_set,
                                 self.test_set, field_name='text', new_field_name='words')

        self.train_set.apply_field(lambda x: len(
            x), field_name='words', new_field_name='seq_len')
        self.val_set.apply_field(lambda x: len(
            x), field_name='words', new_field_name='seq_len')
        self.test_set.apply_field(lambda x: len(
            x), field_name='words', new_field_name='seq_len')
        self.train_set.apply_field(self.find_max_len, field_name='words')

        print(self.max_seq_len)
        self.max_seq_len = min(self.max_seq_len, self.seq_limit)

        self.train_set.apply_field(
            self.seq_regularize, field_name='words', new_field_name='words')
        self.val_set.apply_field(
            self.seq_regularize, field_name='words', new_field_name='words')
        self.test_set.apply_field(
            self.seq_regularize, field_name='words', new_field_name='words')
        # self.train_set.apply(lambda x : text2multi_hot(x['words'],self.vocab_size),new_field_name="input")
        # self.val_set.apply(lambda x : text2multi_hot(x['words'],self.vocab_size),new_field_name='input')
        # self.test_set.apply(lambda x : text2multi_hot(x['words'],self.vocab_size),new_field_name='input')

        # Building target-vector for train_set and test_set.
        print("Building target-vector for train_set and test_set.")
        self.train_set.apply(lambda x: int(
            x['class']), new_field_name="target", is_target=True)
        self.val_set.apply(lambda x: int(
            x['class']), new_field_name="target", is_target=True)
        self.test_set.apply(lambda x: int(
            x['class']), new_field_name="target", is_target=True)
        # self.train_set.apply(lambda x : class2target(x['class'],self.calss_num),new_field_name="target")
        # self.test_set.apply(lambda x : class2target(x['class'],self.calss_num),new_field_name="target")

    def fetch_data(self, path,subset_num=None,us_rate=None,os_rate=None):
        if self.data_src == "all_data":
            # Loading 20newsgroups data and tokenize.
            self.fetch_csv(path,subset_num,us_rate,os_rate)
        else:
            print("No legal data src type:{} ...".format(self.data_src))
            assert(0 == 1)

        self.train_size = self.train_set.get_length()
        self.val_size = self.val_set.get_length()
        self.test_size = self.test_set.get_length()
        return self.train_size, self.val_size, self.test_size


if __name__ == "__main__":
    data_dir = '../../data'
    data = TextData(data_src='all_data')
    # print(data.fetch_data(path=opj(data_dir, 'suboutcome.csv')))
    print(data.fetch_data(path=opj(data_dir, 'essays_outcome.csv'),subset_num=0,os_rate=1.0))
    len_lst = data.train_set.get_field('seq_len')
    plt.hist(len_lst, bins=100)
    plt.title("Essay_Length_distribution")
    # plt.show()
    plt.savefig("Essay_Length_distribution.png")
    print("Test done.")

    print('Saving vocab(TextData)...')
    with open(os.path.join(data_dir, 'vocab','vocab_oversampling.data'), 'wb') as fout:
        pkl.dump(data, fout)
    print('Done with preparing!')
