'''
Test script dor for the Customer Churn Project
A Udacity ML DevOps Engineer Nanodegree Project.

'''


__author__ = "Joeri R. Verbiest"
__copyright__ = "Copyright 2021, Customer Churn Project"
__license__ = "MIT"
__version__ = "1.0"


import os
import logging
import churn_library as cls

import os
os.environ['QT_QPA_PLATFORM']='offscreen'

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        data = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data:"
                      "The file doesn't appear to have rows and columns")
        raise err
    return data


def test_eda(perform_eda, data):
    '''
    test perform eda function
    '''
    perform_eda(data)
    path = "./images/eda"

    # Checking if eda images are available
    try:
        dir_val = os.listdir(path)
        assert dir_val.count('customer_age_distribution.png')
        assert dir_val.count('total_trans_Ct.png')
        assert dir_val.count('churn_distribution.png')
        assert dir_val.count('marital_status_distribution.png')
        assert dir_val.count('heatmap.png')
        logging.info("Testing perform_eda: "
                     "SUCCESS")
    except AssertionError as err:
        logging.warning("Testing perform_eda: "
                        "FAIT, not all images are saved.")
        raise err


def test_encoder_helper(encoder_helper, data):
    '''
    test encoder helper
    '''
    # Checking if cat_columns are available
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    data = encoder_helper(data, cat_columns, 'Churn')

    try:
        for element in cat_columns:
            assert element in data.columns
        logging.info("Testing encoder_helper: "
                     "SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: "
            "FAIL, some missing colmuns.")
        return err

    return data


def test_perform_feature_engineering(perform_feature_engineering, data):
    '''
    test perform_feature_engineering
    '''
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        data, 'Churn')

    try:
        # check shape and length
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        logging.info("Testing perform_feature_engineering: "
                     "SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: "
                      "FAIL, missing output.")
        raise err

    return X_train, X_test, y_train, y_test


def test_train_models(train_models, X_train, X_test, y_train, y_test):
    '''
    test train_models
    '''
    train_models(X_train, X_test, y_train, y_test)

    path = "./images/results/"
    try:
        dir_val = os.listdir(path)
        assert len(dir_val) > 0
    except FileNotFoundError as err:
        logging.error("Testing train_models: "
                      "FAIL, not all result images are saved")
        raise err

    path = "./models/"
    try:
        dir_val = os.listdir(path)
        assert len(dir_val) > 0
        logging.info("Testing train_models: "
                     "SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train_models: "
                      "FAIL, not all models are saved")
        raise err


if __name__ == "__main__":
    DATA = test_import(cls.import_data)
    test_eda(cls.perform_eda, DATA)
    DATA = test_encoder_helper(cls.encoder_helper, DATA)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering(
        cls.perform_feature_engineering, DATA)
    test_train_models(cls.train_models, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
