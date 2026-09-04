# Create Hit Prediction Models

# import libraries
import csv
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
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, ConfusionMatrixDisplay


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


# create train/test split
train_df, test_df = train_test_split(
    bbdf,
    test_size=0.2,
    random_state=100,
    stratify=bbdf['is_hit']
    )

X_tr_wo = train_df[features_without_sprint]
X_tr_with = train_df[features_with_sprint]

X_te_wo = test_df[features_without_sprint]
X_te_with = test_df[features_with_sprint]

y_tr = train_df['is_hit']
y_te = test_df['is_hit']


# plot correlations

# correlation matrix
corr_matrix = train_df[numeric_with_sprint + ['is_hit']].corr()
# print(corr_matrix)

# correlation heatmap
plt.figure(figsize=(8,6))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm'
)

plt.title('Correlation Matrix: Features & Hit Outcome')

# plt.show()


# create models

# logistic regression without sprint
logistic_preprocessor_wo = ColumnTransformer(
    transformers=[
        ('numeric', StandardScaler(), numeric_without_sprint),
        ('categorical', OneHotEncoder(), categorical_features)
    ]
)

logistic_model_wo = Pipeline(
    steps=[
        ('preprocessor', logistic_preprocessor_wo),
        ('model', LogisticRegression())
    ]
)

lmwo_scores = cross_val_score(
    logistic_model_wo,
    X_tr_wo,
    y_tr,
    cv=5,
    scoring='roc_auc'
)
# print(lmwo_scores.mean())

# logistic_model_wo = logistic_model_wo.fit(X_te_wo, y_te)




# logistic model with sprint
logistic_preprocessor_with = ColumnTransformer(
    transformers=[
        ('numeric', StandardScaler(), numeric_with_sprint),
        ('categorical', OneHotEncoder(), categorical_features)
    ]
)

logistic_model_with = Pipeline(
    steps=[
        ('preprocessor', logistic_preprocessor_with),
        ('model', LogisticRegression())
    ]
)

lmw_scores = cross_val_score(
    logistic_model_with,
    X_tr_with,
    y_tr,
    cv=5,
    scoring='roc_auc'
)
# print(lmw_scores.mean())

# logistic_model_with = logistic_model_with.fit(X_te_with, y_te)




# random forest models
def test_random_forest_hyperparameters(numeric_features, X_tr, y_tr):
    """
    Returns best max_depth/n_estimators and corresponding roc_auc score of random forest

    Parameters:
    X_tr (DataFrame)
    y_tr (DataFrame)

    Return:
    best_max_depth (int)
    best_n_estimators (int),
    best_roc_auc (float)
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', 'passthrough', numeric_features),
            ('categorical', OneHotEncoder(), categorical_features)
        ]
    )

    best_max_depth = None
    best_n_estimators = None
    best_roc_auc = None

    for n in [100, 200, 300, 500]:
        for md in [5, 10, 15, 20, None]:
            model = Pipeline(
                steps=[
                    ('preprocessor', preprocessor),
                    ('model', RandomForestClassifier(
                        n_estimators=n,
                        max_depth=md,
                        random_state=100,
                        n_jobs=-1
                    ))
                ]
            )

            scores = cross_val_score(
                model,
                X_tr,
                y_tr,
                cv=5,
                scoring='roc_auc'
            )

            print(
                f"n_estimators={n}, "
                f"max_depth={md}, "
                f"ROC-AUC={scores.mean():.4f}"
            )

            mean_score = scores.mean()

            if best_roc_auc is None:
                best_roc_auc = mean_score
                best_max_depth = md
                best_n_estimators = n
            else:
                if mean_score > best_roc_auc:
                    best_roc_auc = mean_score
                    best_max_depth = md
                    best_n_estimators = n

    return best_max_depth, best_n_estimators, best_roc_auc


# random forest model without sprint

# print(test_random_forest_hyperparameters(numeric_without_sprint, X_tr_wo, y_tr))
# ^^^Returns max_depth=15 and n_estimators=500 (inputting values directly into test model for computational conservation)

# rfwo_preprocessor = ColumnTransformer(
#         transformers=[
#             ('numeric', 'passthrough', numeric_without_sprint),
#             ('categorical', OneHotEncoder(), categorical_features)
#         ]
#     )

# rfwo_model = Pipeline(
#                 steps=[
#                     ('preprocessor', rfwo_preprocessor),
#                     ('model', RandomForestClassifier(
#                         n_estimators=500,
#                         max_depth=15,
#                         random_state=100
#                     ))
#                 ]
#             )

# random_forest_model_wo = rfwo_model.fit(X_te_wo, y_te)


# random forest model with sprint

# print(test_random_forest_hyperparameters(numeric_with_sprint, X_tr_with, y_tr))
# ^^^Returns max_depth=15 and n_estimators=500 (inputting values directly into test model for computational conservation)

# rfwith_preprocessor = ColumnTransformer(
#         transformers=[
#             ('numeric', 'passthrough', numeric_with_sprint),
#             ('categorical', OneHotEncoder(), categorical_features)
#         ]
#     )

# rfwith_model = Pipeline(
#                 steps=[
#                     ('preprocessor', rfwith_preprocessor),
#                     ('model', RandomForestClassifier(
#                         n_estimators=500,
#                         max_depth=15,
#                         random_state=100
#                     ))
#                 ]
#             )

# random_forest_model_with = rfwith_model.fit(X_te_with, y_te)




# XGBoost models
def test_xgboost_hyperparameters(numeric_features, X_tr, y_tr):
    """
    Returns best max_depth/n_estimators and corresponding roc_auc score of xgboost

    Parameters:
    X_tr (DataFrame)
    y_tr (DataFrame)

    Return:
    best_learning_rate (float)
    best_n_estimators (int),
    best_roc_auc (float)
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', 'passthrough', numeric_features),
            ('categorical', OneHotEncoder(), categorical_features)
        ]
    )

    best_learning_rate = None
    best_n_estimators = None
    best_roc_auc = None

    for n in [100, 200, 300, 500]:
        for eta in [0.01, 0.05, 0.1, 0.2]:
            model = Pipeline(
                steps=[
                    ('preprocessor', preprocessor),
                    ('model', XGBClassifier(
                        n_estimators=n,
                        learning_rate=eta,
                        random_state=100,
                        n_jobs=-1
                    ))
                ]
            )

            scores = cross_val_score(
                model,
                X_tr,
                y_tr,
                cv=5,
                scoring='roc_auc'
            )

            print(
                f"n_estimators={n}, "
                f"learning_rate={eta}, "
                f"ROC-AUC={scores.mean():.4f}"
            )

            mean_score = scores.mean()

            if best_roc_auc is None:
                best_roc_auc = mean_score
                best_learning_rate = eta
                best_n_estimators = n
            else:
                if mean_score > best_roc_auc:
                    best_roc_auc = mean_score
                    best_learning_rate = eta
                    best_n_estimators = n

    return best_learning_rate, best_n_estimators, best_roc_auc


# XGBoost model without sprint
# print(test_xgboost_hyperparameters(numeric_without_sprint, X_tr_wo, y_tr))
# ^^^Returns learning_rate=0.05 and n_estimators=500 (inputting values directly into test model for computational conservation)

xgbwo_preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', 'passthrough', numeric_without_sprint),
            ('categorical', OneHotEncoder(), categorical_features)
        ]
    )

xgbwo_model = Pipeline(
                steps=[
                    ('preprocessor', xgbwo_preprocessor),
                    ('model', XGBClassifier(
                        n_estimators=500,
                        learning_rate=0.05,
                        random_state=100
                    ))
                ]
            )

xgboost_model_without = xgbwo_model.fit(X_te_wo, y_te)


# XGBoost model with sprint
print(test_xgboost_hyperparameters(numeric_with_sprint, X_tr_with, y_tr))
# ^^^Returns learning_rate=0.05 and n_estimators=500 (inputting values directly into test model for computational conservation)


xgbwith_preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', 'passthrough', numeric_with_sprint),
            ('categorical', OneHotEncoder(), categorical_features)
        ]
    )

xgbwith_model = Pipeline(
                steps=[
                    ('preprocessor', xgbwith_preprocessor),
                    ('model', XGBClassifier(
                        n_estimators=500,
                        learning_rate=0.05,
                        random_state=100
                    ))
                ]
            )

xgboost_model_with = xgbwith_model.fit(X_te_with, y_te)
