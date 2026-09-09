# Create Hit Type Prediction Models

# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import shap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay


# create DataFrame from batted ball data
bbdf = pd.read_csv('batted_balls.csv')

numeric_without_sprint = [
    'launch_speed',
    'launch_angle',
    'spray_angle'
]

numeric_with_sprint = [
    'launch_speed',
    'launch_angle',
    'spray_angle',
    'sprint_speed'
]

categorical_features = ['bb_type']

features_without_sprint = (
    numeric_without_sprint + categorical_features
)

features_with_sprint = (
    numeric_with_sprint + categorical_features
)

bbdf = bbdf.dropna(subset=features_with_sprint)

# create outcomes column
outcomes = ['out', 'single', 'double', 'triple', 'home_run']

bbdf['outcome'] = np.where(
    bbdf['events'].isin(outcomes),
    bbdf['events'],
    'out'
)
