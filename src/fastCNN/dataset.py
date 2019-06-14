import os
import string
import time
import math
import os.path as op
from os.path import join as opj
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
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
    new_data = []
    for i in range(len(data)):
        text = data[i]
        if text == "":
            print("Empty text.{0}".format(i))
        elif text is None:
            print("None type text.")
        elif not isinstance(text,str):
            print("ATTENTION TYPE:{}".format(text))
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
    test_set = DataSet()
    val_set = DataSet()
    train_size = 0
    val_size = 0
    test_size = 0

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

    def fetch_csv(self, path=None):
        print("Loading data from {} .......".format(path))
        df = pd.read_csv(path, header=0)

        train_df = df[df['split'] == 'train']
        val_df = df[df['split'] == 'val']
        test_df = df[df['split'] == 'test']

        # text_vars=["title", "short_description", "need_statement", "essay"]
        text_vars = "essay"  # only select the essay column
        print("tokenize train set")
        train_input = tokenize(train_df[text_vars].values)
        print("tokenize val set")
        val_input = tokenize(val_df[text_vars].values)
        print("tokenize test set")
        test_input = tokenize(test_df[text_vars].values)

        target_var = "y"
        train_target = train_df[target_var].values
        val_target = val_df[target_var].values
        test_target = test_df[target_var].values

        assert (self.class_num == 2)

        # Building Fastnlp dataset.
        print("Building Fastnlp dataset.")
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

    def fetch_data(self, path=None):
        if self.data_src == "all_data":
            # Loading 20newsgroups data and tokenize.
            self.fetch_csv(path)
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
    print(data.fetch_data(path=opj(data_dir, 'suboutcome.csv')))
    len_lst = data.train_set.get_field('seq_len')
    plt.hist(len_lst, bins=500)
    plt.title("Essay_Length_distribution")
    # plt.show()
    plt.savefig("Essay_Length_distribution.png")
    print("Test done.")

    print('Saving vocab(TextData)...')
    with open(os.path.join(data_dir, 'vocab.data'), 'wb') as fout:
        pkl.dump(data, fout)
    print('Done with preparing!')
