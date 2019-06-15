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

data_dir = "../../data"
essays_file = opj(data_dir, "essays.csv")
outcome_file = opj(data_dir, "outcomes.csv")
projects_file = opj(data_dir, "projects.csv")
combined_file = opj(data_dir, "essays_all_outcome.csv")

print ("Getting essay features...")
df = pd.read_csv(essays_file)
outcomes_df = pd.read_csv(outcome_file)
projects_df = pd.read_csv(projects_file)
print("Merging on projectid")
df = pd.merge(df, outcomes_df, how = 'left', on = 'projectid')
df = pd.merge(df, projects_df, how = 'inner', on = 'projectid')
df["split"] = "train"
df["split"][df["date_posted"]<"2010-04-01"] = "none"
df["split"][df["date_posted"]>="2013-01-01"] = "val"
df["split"][df["date_posted"]>="2014-01-01"] = "test"
print("Test size:{}".format(len(df[df["split"]=="test"] )))

df = df[df["split"]!="none"]
# df["is_exciting"][ (df["is_exciting"] != "t") & (df["is_exciting"] != "f") ] = "f"
# df["y"] = 0
# df["y"][df["is_exciting"]=="t"] = 1
tf_vars = [  "is_exciting ","at_least_1_teacher_referred_donor","great_chat","fully_funded",
                "at_least_1_green_donation","donation_from_thoughtful_donor",
                "three_or_more_non_teacher_referred_donors",
                "one_non_teacher_referred_donor_giving_100_plus"]
for var in tf_vars:
    df[var][ (df[var] != "t") & (df[var] != "f") ] = "f"

text_vars=["title", "short_description", "need_statement", "essay"]
df = df[ ["projectid","split"] + text_vars + tf_vars ]

df.to_csv(combined_file,index = False)
print("Data saved to file {}".format(combined_file))
print("Done.")
