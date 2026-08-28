# Perform EDA

# import libraries
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# create DataFrame from batted ball data
bbdf = pd.read_csv("batted_balls.csv")

features = [
    "launch_speed",
    "launch_angle",
    "spray_angle",
    "bb_type"
]


# describe numerical features
print(bbdf[features].describe(include=[np.number]).round(2))

# describe categorical features
print(bbdf[features + ["events"]].describe(include=[object]))
