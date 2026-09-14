# Create Hit Type Prediction Models

# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
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


# create train/test split
train_df, test_df = train_test_split(
    bbdf,
    test_size=0.2,
    random_state=100,
    stratify=bbdf['outcome']
    )

X_tr_wo = train_df[features_without_sprint]
X_tr_with = train_df[features_with_sprint]

X_te_wo = test_df[features_without_sprint]
X_te_with = test_df[features_with_sprint]

y_tr = train_df['outcome']
y_te = test_df['outcome']

label_encoder = LabelEncoder()
y_tr_xgb = label_encoder.fit_transform(y_tr)


# plot correlations

# correlation matrix
outcome_dummies = pd.get_dummies(
    train_df['outcome'],
    prefix='outcome',
    dtype=int
)

corr_data = pd.concat(
    [
        train_df[numeric_with_sprint],
        outcome_dummies
    ],
    axis=1
)

corr_matrix = corr_data.corr()


# correlation heatmap

outcome_columns = outcome_dummies.columns

outcome_corr = corr_matrix.loc[
    numeric_with_sprint,
    outcome_columns
]

# plt.figure(figsize=(8, 5))

# sns.heatmap(
#     outcome_corr,
#     annot=True,
#     cmap="coolwarm",
#     fmt=".2f",
#     vmin=-1,
#     vmax=1
# )

# plt.title("Feature Correlations with Hit Outcomes")

# plt.show()




# create models

# logistic models

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
        ('model', LogisticRegression(max_iter=1000))
    ]
)

lmwo_scores = cross_val_score(
    logistic_model_wo,
    X_tr_wo,
    y_tr,
    cv=5,
    scoring='f1_macro'
)
# print(lmwo_scores.mean())

logistic_model_wo = logistic_model_wo.fit(X_tr_wo, y_tr)


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
        ('model', LogisticRegression(max_iter=1000))
    ]
)

lmw_scores = cross_val_score(
    logistic_model_with,
    X_tr_with,
    y_tr,
    cv=5,
    scoring='f1_macro'
)
# print(lmw_scores.mean())

logistic_model_with = logistic_model_with.fit(X_tr_with, y_tr)




# random forest models
def test_random_forest_hyperparameters(numeric_features, X_tr, y_tr):
    """
    Returns best max_depth/n_estimators and corresponding f1 macro score of random forest

    Parameters:
    X_tr (DataFrame)
    y_tr (DataFrame)

    Return:
    best_max_depth (int)
    best_n_estimators (int),
    best_f1_macro (float)
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', 'passthrough', numeric_features),
            ('categorical', OneHotEncoder(), categorical_features)
        ]
    )

    best_max_depth = None
    best_n_estimators = None
    best_f1_macro = None

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
                scoring='f1_macro'
            )

            print(
                f"n_estimators={n}, "
                f"max_depth={md}, "
                f"f1 Macro={scores.mean():.4f}"
            )

            mean_score = scores.mean()

            if best_f1_macro is None:
                best_f1_macro = mean_score
                best_max_depth = md
                best_n_estimators = n
            else:
                if mean_score > best_f1_macro:
                    best_f1_macro = mean_score
                    best_max_depth = md
                    best_n_estimators = n

    return best_max_depth, best_n_estimators, best_f1_macro


# random forest without sprint

# print(test_random_forest_hyperparameters(numeric_without_sprint, X_tr_wo, y_tr))
# ^^^Returns max_depth=15 and n_estimators=500 (inputting values directly into test model for computational conservation)

rfwo_preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', 'passthrough', numeric_without_sprint),
            ('categorical', OneHotEncoder(), categorical_features)
        ]
    )

rfwo_model = Pipeline(
                steps=[
                    ('preprocessor', rfwo_preprocessor),
                    ('model', RandomForestClassifier(
                        n_estimators=500,
                        max_depth=15,
                        random_state=100
                    ))
                ]
            )

random_forest_model_wo = rfwo_model.fit(X_tr_wo, y_tr)


# random forest with sprint

# print(test_random_forest_hyperparameters(numeric_with_sprint, X_tr_with, y_tr))
# ^^^Returns max_depth=20 and n_estimators=500 (inputting values directly into test model for computational conservation)

rfwith_preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', 'passthrough', numeric_with_sprint),
            ('categorical', OneHotEncoder(), categorical_features)
        ]
    )

rfwith_model = Pipeline(
                steps=[
                    ('preprocessor', rfwith_preprocessor),
                    ('model', RandomForestClassifier(
                        n_estimators=500,
                        max_depth=15,
                        random_state=100
                    ))
                ]
            )

random_forest_model_with = rfwith_model.fit(X_tr_with, y_tr)




# XGBoost models
def test_xgboost_hyperparameters(numeric_features, X_tr, y_tr):
    """
    Returns best max_depth/n_estimators and corresponding f1 macro score of xgboost

    Parameters:
    X_tr (DataFrame)
    y_tr (DataFrame)

    Return:
    best_learning_rate (float)
    best_n_estimators (int),
    best_f1_macro (float)
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', 'passthrough', numeric_features),
            ('categorical', OneHotEncoder(), categorical_features)
        ]
    )

    best_learning_rate = None
    best_n_estimators = None
    best_f1_macro = None

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
                scoring='f1_macro'
            )

            print(
                f"n_estimators={n}, "
                f"learning_rate={eta}, "
                f"f1 Macro={scores.mean():.4f}"
            )

            mean_score = scores.mean()

            if best_f1_macro is None:
                best_f1_macro = mean_score
                best_learning_rate = eta
                best_n_estimators = n
            else:
                if mean_score > best_f1_macro:
                    best_f1_macro = mean_score
                    best_learning_rate = eta
                    best_n_estimators = n

    return best_learning_rate, best_n_estimators, best_f1_macro


# XGBoost model without sprint
# print(test_xgboost_hyperparameters(numeric_without_sprint, X_tr_wo, y_tr_xgb))
# ^^^Returns learning_rate=0.2 and n_estimators=100 (inputting values directly into test model for computational conservation)

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
                        n_estimators=100,
                        learning_rate=0.2,
                        random_state=100
                    ))
                ]
            )

xgboost_model_wo = xgbwo_model.fit(X_tr_wo, y_tr_xgb)


# XGBoost model with sprint
print(test_xgboost_hyperparameters(numeric_with_sprint, X_tr_with, y_tr_xgb))
# ^^^Returns learning_rate=0.2 and n_estimators=200 (inputting values directly into test model for computational conservation)

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

xgboost_model_with = xgbwith_model.fit(X_tr_with, y_tr)




# evaluate/compare performance of all models

# metrics
def evaluate_model(model, X_te, y_te):
    """
    Returns roc-auc, accuracy, precision, recall, f1, and confusion matrix for model

    Parameters:
    model
    X_te (DataFrame)
    y_te (Series)

    Return:
    f1_macro (float)
    f1_weighted (float)
    accuracy (float)
    precision (float)
    recall (float)
    conf_mat (confusion_matrix)
    """
    preds = model.predict(X_te)

    f1_macro = f1_score(y_te, preds, average='macro')
    f1_weighted = f1_score(y_te, preds, average='weighted')
    accuracy = accuracy_score(y_te, preds)
    precision = precision_score(y_te, preds)
    recall = recall_score(y_te, preds)
    conf_mat = confusion_matrix(y_te, preds)

    return f1_macro, f1_weighted, accuracy, precision, recall, conf_mat
