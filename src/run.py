import sys
import os
import pickle
import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser('KDD CUP 2014')

    parser.add_argument('--model',choices=['LSTM'],default='')