# Perform EDA

# import libraries
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# create DataFrame from batted ball data
bbdf = pd.read_csv("batted_balls.csv")
print(bbdf)
