# Perform EDA

# import libraries
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# load batted ball data
def load_batted_ball_data(filename):
    """
    Loads batted ball data from a CSV file.

    Parameter:
    filename (str): Path to the CSV file containing batted ball data.

    Returns:
    data_list (list[dict]): List of dictionairies with batted ball data.
    """
    data_list = []
    with open(filename, 'r', encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data_list += [row]
    return data_list

bbs = load_batted_ball_data("batted_balls.csv")
