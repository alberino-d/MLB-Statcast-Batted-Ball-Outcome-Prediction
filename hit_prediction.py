# Create Hit Prediction Models

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
# plt.figure(figsize=(8,6))

# sns.heatmap(
#     corr_matrix,
#     annot=True,
#     cmap='coolwarm'
# )

# plt.title('Correlation Matrix: Features & Hit Outcome')

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

logistic_model_with = logistic_model_with.fit(X_tr_with, y_tr)




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


# random forest model with sprint

# print(test_random_forest_hyperparameters(numeric_with_sprint, X_tr_with, y_tr))
# ^^^Returns max_depth=15 and n_estimators=500 (inputting values directly into test model for computational conservation)

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

xgboost_model_wo = xgbwo_model.fit(X_tr_wo, y_tr)


# XGBoost model with sprint
# print(test_xgboost_hyperparameters(numeric_with_sprint, X_tr_with, y_tr))
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

xgboost_model_with = xgbwith_model.fit(X_tr_with, y_tr)




# evaluate/compare performance of all models

# metrics
def evaluate_model(model, X_te, y_te):
    """
    Returns .......

    Parameters:
    model
    X_te (DataFrame)
    y_te (Series)

    Return:
    roc_auc (float)
    accuracy (float)
    precision (float)
    recall (float)
    f1 (float)
    conf_mat (confusion_matrix)
    """
    preds = model.predict(X_te)
    probas = model.predict_proba(X_te)[:,1]

    roc_auc = roc_auc_score(y_te, probas)
    accuracy = accuracy_score(y_te, preds)
    precision = precision_score(y_te, preds)
    recall = recall_score(y_te, preds)
    f1= f1_score(y_te, preds)
    conf_mat = confusion_matrix(y_te, preds)

    return roc_auc, accuracy, precision, recall, f1, conf_mat

# logistic regression without sprint
# lrwo_roc_auc, lrwo_accuracy, lrwo_precision, lrwo_recall, lrwo_f1, lrwo_conf_mat = evaluate_model(logistic_model_wo, X_te_wo, y_te)
# print('Logistic Regression w/o Sprint Performance:')
# print(f'Roc-Auc Score: {lrwo_roc_auc:.4f}')
# print(f'Accuracy Score: {lrwo_accuracy:.4f}')
# print(f'Precision Score: {lrwo_precision:.4f}')
# print(f'Recall Score: {lrwo_recall:.4f}')
# print(f'f1 Score: {lrwo_f1:.4f}')
# ConfusionMatrixDisplay(lrwo_conf_mat, display_labels=['Out', 'Hit']).plot(cmap='Blues')
# plt.show()

# logistic regression with sprint
# lrwith_roc_auc, lrwith_accuracy, lrwith_precision, lrwith_recall, lrwith_f1, lrwith_conf_mat = evaluate_model(logistic_model_with, X_te_with, y_te)
# print('Logistic Regression w/ Sprint Performance:')
# print(f'Roc-Auc Score: {lrwith_roc_auc:.4f}')
# print(f'Accuracy Score: {lrwith_accuracy:.4f}')
# print(f'Precision Score: {lrwith_precision:.4f}')
# print(f'Recall Score: {lrwith_recall:.4f}')
# print(f'f1 Score: {lrwith_f1:.4f}')
# ConfusionMatrixDisplay(lrwith_conf_mat, display_labels=['Out', 'Hit']).plot(cmap='Blues')
# plt.show()

# random forest without sprint
# rfwo_roc_auc, rfwo_accuracy, rfwo_precision, rfwo_recall, rfwo_f1, rfwo_conf_mat = evaluate_model(random_forest_model_wo, X_te_wo, y_te)
# print('Random Forest w/o Sprint Performance:')
# print(f'Roc-Auc Score: {rfwo_roc_auc:.4f}')
# print(f'Accuracy Score: {rfwo_accuracy:.4f}')
# print(f'Precision Score: {rfwo_precision:.4f}')
# print(f'Recall Score: {rfwo_recall:.4f}')
# print(f'f1 Score: {rfwo_f1:.4f}')
# ConfusionMatrixDisplay(rfwo_conf_mat, display_labels=['Out', 'Hit']).plot(cmap='Blues')
# plt.show()

# random forest with sprint
# rfwith_roc_auc, rfwith_accuracy, rfwith_precision, rfwith_recall, rfwith_f1, rfwith_conf_mat = evaluate_model(random_forest_model_with, X_te_with, y_te)
# print('Random Forest w/ Sprint Performance:')
# print(f'Roc-Auc Score: {rfwith_roc_auc:.4f}')
# print(f'Accuracy Score: {rfwith_accuracy:.4f}')
# print(f'Precision Score: {rfwith_precision:.4f}')
# print(f'Recall Score: {rfwith_recall:.4f}')
# print(f'f1 Score: {rfwith_f1:.4f}')
# ConfusionMatrixDisplay(rfwith_conf_mat, display_labels=['Out', 'Hit']).plot(cmap='Blues')
# plt.show()

# XGBoost without sprint
# xgbwo_roc_auc, xgbwo_accuracy, xgbwo_precision, xgbwo_recall, xgbwo_f1, xgbwo_conf_mat = evaluate_model(xgboost_model_wo, X_te_wo, y_te)
# print('XGBoost w/o Sprint Performance:')
# print(f'Roc-Auc Score: {xgbwo_roc_auc:.4f}')
# print(f'Accuracy Score: {xgbwo_accuracy:.4f}')
# print(f'Precision Score: {xgbwo_precision:.4f}')
# print(f'Recall Score: {xgbwo_recall:.4f}')
# print(f'f1 Score: {xgbwo_f1:.4f}')
# ConfusionMatrixDisplay(xgbwo_conf_mat, display_labels=['Out', 'Hit']).plot(cmap='Blues')
# plt.show()

# XGBoost with sprint
# xgbwith_roc_auc, xgbwith_accuracy, xgbwith_precision, xgbwith_recall, xgbwith_f1, xgbwith_conf_mat = evaluate_model(xgboost_model_with, X_te_with, y_te)
# print('XGBoost w/ Sprint Performance:')
# print(f'Roc-Auc Score: {xgbwith_roc_auc:.4f}')
# print(f'Accuracy Score: {xgbwith_accuracy:.4f}')
# print(f'Precision Score: {xgbwith_precision:.4f}')
# print(f'Recall Score: {xgbwith_recall:.4f}')
# print(f'f1 Score: {xgbwith_f1:.4f}')
# ConfusionMatrixDisplay(xgbwith_conf_mat, display_labels=['Out', 'Hit']).plot(cmap='Blues')
# plt.show()




# evaluate best model mechanics

# best logistic regression model
# log_preprocessor = logistic_model_with.named_steps['preprocessor']
# log_model = logistic_model_with.named_steps['model']
# log_feat_names = log_preprocessor.get_feature_names_out()

# print("Logistic Model w/ Sprint Coefficients:")
# for i, feature in enumerate(log_feat_names):
#     print(f"{feature}: {log_model.coef_[0, i]:.2f}")
# print(f"b: {log_model.intercept_[0]:.2f}")


# best random forest model
# rf_preprocessor = random_forest_model_wo.named_steps['preprocessor']
# rf_model = random_forest_model_wo.named_steps ['model']
# rf_feat_names = rf_preprocessor.get_feature_names_out()

# rf_feat_import = rf_model.feature_importances_

# idx = np.argsort(rf_feat_import).astype(int)
# rf_feat_list = [rf_feat_names[i] for i in idx][::-1]
# rf_feat_import = rf_feat_import[idx][::-1]
    
# sns.barplot(x=rf_feat_import, y=rf_feat_list, color='lightblue', edgecolor='black')
# plt.xlabel('Feature Importance')

# plt.show()


# best XGBoost model
# def create_shap_graph(model, X_tr):
#     """
#     Prints SHAP model

#     Parameters:
#     model (Pipeline)
#     X_tr (DataFrame)
#     """
#     preprocessor = model.named_steps['preprocessor']

#     model = model.named_steps['model']

#     X_tr_transformed = preprocessor.transform(X_tr)

#     feature_names = preprocessor.get_feature_names_out()

#     explainer = shap.TreeExplainer(model)

#     shap_values = explainer.shap_values(X_tr_transformed)

#     shap.summary_plot(shap_values, X_tr_transformed, feature_names=feature_names)

# create_shap_graph(xgboost_model_with, X_tr_with)
