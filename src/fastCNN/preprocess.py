import os
import string
import time
import math
import os.path as op
from os.path import join as opj

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

data_dir = "KDD2014\data"
essays_file = opj(data_dir, "essays.csv")
outcome_file = opj(data_dir, "outcomes.csv")
projects_file = opj(data_dir, "projects.csv")
combined_file = opj(data_dir, "essays_outcome.csv")

print ("Getting essay features...")
df = pd.read_csv(essays_file)
outcomes_df = pd.read_csv(outcome_file)
projects_df = pd.read_csv(projects_file)

df = pd.merge(df, outcomes_df, how = 'left', on = 'projectid')
df = pd.merge(df, projects_df, how = 'inner', on = 'projectid')
df["split"] = "train"
df["split"][df["date_posted"]<"2010-04-01"] = "none"
df["split"][df["date_posted"]>="2013-01-01"] = "val"
df["split"][df["date_posted"]>="2014-01-01"]= "test"
df = df[df["split"]!="none"]
df["y"] = 0
df["y"][df["is_exciting"]=="t"] = 1
text_vars=["title", "short_description", "need_statement", "essay"]
df = df[text_vars+["split","is_exciting","projectid"]]
df.to_csv(combined_file, index=False)
print("Done.")
