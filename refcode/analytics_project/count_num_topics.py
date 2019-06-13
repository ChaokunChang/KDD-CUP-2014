# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 18:47:15 2016

@author: bbux
"""

import pandas as pd
import os
import operator
import sys

"""
Start Script
"""
def count_non_zero(row):
    return len([x for x in row.values if x > 0])
    
inputfile = sys.argv[1]
if not os.path.exists(inputfile):
    raise Exception("Input file does not exist!: " + inputfile)

df = pd.read_csv(inputfile, index_col=0)
df = df.drop('projectid', 1)
counts = {}
for col in df.apply(count_non_zero, axis=1):
    if col in counts:
        counts[col] += 1
    else:
        counts[col] = 1

for k, v in sorted(counts.items(), key=operator.itemgetter(1)):
    print("%s, %d" % (k, v))

   
