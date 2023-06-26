import itertools as it

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


def x_y_split(data_set, target, features):
    X = data_set[features]
    y = data_set[target]
    return X, y



def get_col_combos(cols, min_combo_len=1):
    '''
    Takes in list of columns
    Returns a list of all possible column combinations
    '''
    column_combinations = []
    for r in range(min_combo_len, len(cols)+1):
        for combo in list(it.combinations(cols, r)):
            column_combinations.append(list(combo))
    
    return column_combinations

def run_baseline_model(train, validate, target):
    
    # split the data
    X_train, y_train = x_y_split(data_set=train, target=target, features=train.columns)

    X_validate, y_validate = x_y_split(data_set=validate, target=target, features=validate.columns)
    
    # Make Dummy classifier
    clf = DummyClassifier(strategy='most_frequent')
    
    # Fit Dummy classifier
    clf.fit(X_train, y_train)
    
    # Predict
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_validate)

    scores = []

    output = {
                'features': list(train.columns),
                'model': 'baseline',
                'train_accuracy': metrics.accuracy_score(y_train, y_train_pred),
                'train_precision': metrics.precision_score(y_train, y_train_pred),
                'train_recall/TPR': metrics.recall_score(y_train, y_train_pred),
                'train_f1': metrics.f1_score(y_train, y_train_pred),
                'validate_accuracy':  metrics.accuracy_score(y_validate, y_val_pred),
                'validate_precision': metrics.precision_score(y_validate, y_val_pred),
                'validate_recall/TPR': metrics.recall_score(y_validate, y_val_pred),
                'validate_f1': metrics.f1_score(y_validate, y_val_pred),
                }
    scores.append(output)
    
    scores_df = pd.DataFrame(scores)
    
    return scores_df

def run_lr_models(train, validate, target, feature_combinations, random_state=None):   
    '''
    takes in train set, validate set, target, columns to model
    gets all column combinations and runs knn models for each combination
    hyperparameter, k, is tuned by running each k in specified range, k_range
    scores for each model are stored in a DataFrame, scores_df
    
    Returns scores_df, a DataFrame of model scores
    '''

    # for each combo of features
    for i, features in enumerate(feature_combinations):
        # split X and y
        X_train, y_train = x_y_split(data_set=train, target=target, features=features)

        X_validate, y_validate = x_y_split(data_set=validate, target=target, features=features)
        
        scores = []
        
        # hyperparameter tuning
        for c in [1*10**x for x in range(-3, 4)]:
            # Make KNN classifier 
            clf = LogisticRegression(C=c, max_iter=150, random_state=125)

            # Fit KNN classifier
            clf.fit(X_train, y_train)

            # Predict
            y_pred_train = clf.predict(X_train)
            y_pred_val = clf.predict(X_validate)

            output = {
                'features':features,
                'model': f'LogReg: C={c}',
                'train_accuracy': metrics.accuracy_score(y_train, y_pred_train),
                'train_precision': metrics.precision_score(y_train, y_pred_train),
                'train_recall/TPR': metrics.recall_score(y_train, y_pred_train),
                'train_f1': metrics.f1_score(y_train, y_pred_train),
                'validate_accuracy':  metrics.accuracy_score(y_validate, y_pred_val),
                'validate_precision': metrics.precision_score(y_validate, y_pred_val),
                'validate_recall/TPR': metrics.recall_score(y_validate, y_pred_val),
                'validate_f1': metrics.f1_score(y_validate, y_pred_val),
            }
            scores.append(output)
            
        # initialize scores DataFrame for first iteration
        if i == 0:
            scores_df = pd.DataFrame(scores)
        # concat scores DataFrames for the rest of the iterations
        else:
            scores_df = pd.concat([scores_df, pd.DataFrame(scores)])
            
    return scores_df

def run_knn_models(train, validate, target, feature_combinations, k_range=[2, 10, 15, 20, 25, 40, 70, 100, 150]):   
    '''
    takes in train set, validate set, target, columns to model
    gets all column combinations and runs knn models for each combination
    hyperparameter, k, is tuned by running each k in specified range, k_range
    scores for each model are stored in a DataFrame, scores_df
    
    Returns scores_df, a DataFrame of model scores
    '''

    # for each combo of features
    for i, features in enumerate(feature_combinations):
        # split X and y
        X_train, y_train = x_y_split(data_set=train, target=target, features=features)

        X_validate, y_validate = x_y_split(data_set=validate, target=target, features=features)
        
        scores = []
        
        # hyperparameter tuning
        for k in k_range:
            # Make KNN classifier 
            knn = KNeighborsClassifier(n_neighbors=k)

            # Fit KNN classifier
            knn.fit(X_train, y_train)

            # Predict
            y_pred_train = knn.predict(X_train)
            y_pred_val = knn.predict(X_validate)

            output = {
                'features': features,
                'model': f'KNN, k={k}',
                'train_accuracy': metrics.accuracy_score(y_train, y_pred_train),
                'train_precision': metrics.precision_score(y_train, y_pred_train),
                'train_recall/TPR': metrics.recall_score(y_train, y_pred_train),
                'train_f1': metrics.f1_score(y_train, y_pred_train),
                'validate_accuracy':  metrics.accuracy_score(y_validate, y_pred_val),
                'validate_precision': metrics.precision_score(y_validate, y_pred_val),
                'validate_recall/TPR': metrics.recall_score(y_validate, y_pred_val),
                'validate_f1': metrics.f1_score(y_validate, y_pred_val),
            }
            scores.append(output)
            
        # initialize scores DataFrame for first iteration
        if i == 0:
            scores_df = pd.DataFrame(scores)
        # concat scores DataFrames for the rest of the iterations
        else:
            scores_df = pd.concat([scores_df, pd.DataFrame(scores)])
            
    return scores_df

def run_lr_models(train, validate, target, feature_combinations, random_state=None):   
    '''
    takes in train set, validate set, target, columns to model
    gets all column combinations and runs knn models for each combination
    hyperparameter, k, is tuned by running each k in specified range, k_range
    scores for each model are stored in a DataFrame, scores_df
    
    Returns scores_df, a DataFrame of model scores
    '''

    # for each combo of features
    for i, features in enumerate(feature_combinations):
        # split X and y
        X_train, y_train = x_y_split(data_set=train, target=target, features=features)

        X_validate, y_validate = x_y_split(data_set=validate, target=target, features=features)
        
        scores = []
        
        # hyperparameter tuning
        for c in [1*10**x for x in range(-3, 4)]:
            # Make KNN classifier 
            clf = LogisticRegression(C=c, max_iter=150, random_state=125)

            # Fit KNN classifier
            clf.fit(X_train, y_train)

            # Predict
            y_pred_train = clf.predict(X_train)
            y_pred_val = clf.predict(X_validate)

            output = {
                'features':features,
                'model': f'LogReg: C={c}',
                'train_accuracy': metrics.accuracy_score(y_train, y_pred_train),
                'train_precision': metrics.precision_score(y_train, y_pred_train),
                'train_recall/TPR': metrics.recall_score(y_train, y_pred_train),
                'train_f1': metrics.f1_score(y_train, y_pred_train),
                'validate_accuracy':  metrics.accuracy_score(y_validate, y_pred_val),
                'validate_precision': metrics.precision_score(y_validate, y_pred_val),
                'validate_recall/TPR': metrics.recall_score(y_validate, y_pred_val),
                'validate_f1': metrics.f1_score(y_validate, y_pred_val),
            }
            scores.append(output)
            
        # initialize scores DataFrame for first iteration
        if i == 0:
            scores_df = pd.DataFrame(scores)
        # concat scores DataFrames for the rest of the iterations
        else:
            scores_df = pd.concat([scores_df, pd.DataFrame(scores)])
            
    return scores_df

def plot_DT_feature_importances(train, validate, target, features, max_depth):
    # split X and y
    X_train, y_train = x_y_split(data_set=train, target=target, features=features)

    X_validate, y_validate = x_y_split(data_set=validate, target=target, features=features)

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=125, class_weight=None)

    # Fit DT classifier
    clf.fit(X_train, y_train)

    importances = clf.feature_importances_
    feature_importances = pd.DataFrame({'Feature': features, 'Importance': importances})

    # sort df by feature importance
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    # plot
    plt.figure(figsize=(4,2))
    sns.barplot(x='Importance', y='Feature', data=feature_importances)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances')
    plt.show()