'''
A library of functions for the Customer Churn Project
A Udacity ML DevOps Engineer Nanodegree Project.

'''

__author__ = "Joeri R. Verbiest"
__copyright__ = "Copyright 2021, Customer Churn Project"
__license__ = "MIT"
__version__ = "1.0"

import logging
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import os
os.environ['QT_QPA_PLATFORM']='offscreen'

# display message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data: pandas dataframe
    '''

    data = pd.read_csv(pth)

    return data


def perform_eda(data):
    '''
    perform eda on df and save figures to images folder
    input:
            data: pandas dataframe

    output:
            None
    '''
    data['Churn'] = data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Churn distribution
    plt.figure(figsize=(20, 10))
    data['Churn'].hist()
    plt.title("Churn distribution")
    plt.savefig('./images/eda/churn_distribution.png')
    plt.close()

    # Customer age distribution
    plt.figure(figsize=(20, 10))
    data['Customer_Age'].hist()
    plt.title("Customer age distribution")
    plt.savefig('./images/eda/customer_age_distribution.png')
    plt.close()

    # Marital status distribution
    plt.figure(figsize=(20, 10))
    data.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.title("Marital status distribution")
    plt.savefig('./images/eda/marital_status_distribution.png')
    plt.close()

    # Total Trans Ct distribution
    plt.figure(figsize=(20, 10))
    sns.distplot(data['Total_Trans_Ct'])
    plt.title(" Total Trans Ct distribution")
    plt.savefig('./images/eda/total_trans_Ct.png')
    plt.close()

    # Heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(data.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.title("Heatmap")
    plt.savefig('./images/eda/heatmap.png')
    plt.close()


def encoder_helper(data, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for
                      naming variables or index y column]

    output:
            data: pandas dataframe with new columns for
    '''
    for element in category_lst:
        lst = []
        group = data.groupby(element).mean()[response]

        for val in data[element]:
            lst.append(group.loc[val])

        name = element + '_' + response
        data[name] = lst

    return data


def perform_feature_engineering(data, response):
    '''
    input:
              data: pandas dataframe
              response: string of response name [optional argument that could be used for naming
                        variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    y = data['Churn']
    X = pd.DataFrame()
    data = encoder_helper(data, cat_columns, response)
    X[keep_cols] = data[keep_cols]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # Random Forest
    plt.figure()
    plt.rc('figure', figsize=(20, 10))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_train, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/rf_results.png')
    plt.close()

    # Logistic Regression
    plt.figure()
    plt.rc('figure', figsize=(20, 10))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_train, y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
        'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/logistic_results.png')
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importance
    names = [X_data.columns[i] for i in indices]

    # Create plot and save
    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # store roc curve with score
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    _ = plot_roc_curve(cv_rfc.best_estimator_,
                       X_test, y_test, ax=axis, alpha=0.8)
    lrc_plot.plot(ax=axis, alpha=0.8)
    plt.savefig('./images/results/roc_curve_result.png')
    plt.close()
    logger.info("Save best model.")
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # store model results
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    # store feature importances plot
    feature_importance_plot(cv_rfc.best_estimator_, X_train,
                            './images/results/feature_importances.png')


if __name__ == "__main__":
    logger.info("Import data.")
    BANK_DATA = import_data("./data/bank_data.csv")

    logger.info("Perform EDA.")
    perform_eda(BANK_DATA)

    logger.info("Split Train & Test data.")
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        BANK_DATA, 'Churn')

    logger.info("Train and store model en results.")
    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
