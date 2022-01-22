"""Diagnostics
"""
import timeit
import os
import json
import pickle
import subprocess
import sys
import pandas as pd

# Load config.json and get environment variables
with open('config.json', 'r') as json_file:
    config = json.load(json_file)
json_file.close()


def model_predictions(data_path, data_filename, model_path, model_filename):
    """Function to get model predictions
    """
    data = pd.read_csv(os.path.join(data_path, data_filename))
    y = data['exited']
    data = data.drop(['corporation', 'exited'], axis=1)

    with open(os.path.join(model_path, model_filename), 'rb') as pickle_file:
        model = pickle.load(pickle_file)
    pickle_file.close()

    model_predict = model.predict(data)

    return model, model_predict, y, data


def dataframe_summary():
    """Function to get summary statistics
    """
    data = pd.read_csv(config['finaldata_filename'])
    data = data.drop(['corporation', 'exited'], axis=1)

    statistics = []
    for column in data.columns:
        statistics.append([column + " (mean):", data[column].mean()])
        statistics.append([column + " (median):", data[column].median()])
        statistics.append(
            [column + " (standard deviation):", data[column].std()])

    return statistics


def dataframe_missing_data(data_filename):
    """Calculate missing data
    """
    data = pd.read_csv(data_filename)

    missing_data = []
    for column in data.columns:
        missing_data.append(
            [column + " (%):", int(data[column].isna().sum() / data[column].shape[0] * 100)])

    return missing_data


def execution_time():
    """Function to get timings
    """
    starttime = timeit.default_timer()
    _ = subprocess.run(['python', 'ingestion.py'], capture_output=True)
    timing = timeit.default_timer() - starttime

    return round(timing, 2)


def outdated_packages_list():
    """Function to check dependencies
    """
    outdated_packages = subprocess.check_output(
        ['pip', 'list', '--outdated']).decode(sys.stdout.encoding)

    return outdated_packages


if __name__ == '__main__':
    model_pth = os.path.join(config['prod_deployment_path'])
    model_file = os.path.join(config['trainedmodel_filename'])
    data_file = os.path.join(config['finaldata_filename'])

    model_predictions("", data_file, model_pth, model_file)
    print(dataframe_summary())
    print(dataframe_missing_data(data_file))
    print(execution_time())
    print(outdated_packages_list())
