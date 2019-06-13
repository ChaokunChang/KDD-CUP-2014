import os
import cv2
import numpy as np
import torch
from numpy.random import RandomState
from torch.utils.data import Dataset
from torch.autograd import Variable
import os.path as op
from os.path import join as opj
import pandas as pd

class EssayData(Dataset):
    def __init__(self, data_dir='../../data', data_file='preprocessall.csv' ):
        super(EssayData,self).__init__()
        self.df = pd.read_csv(opj(data_dir,data_file))

    def __len__(self):
        return len(self.df)

    def __getitem__(self,index):
        return np.array(self.df.loc[index])
