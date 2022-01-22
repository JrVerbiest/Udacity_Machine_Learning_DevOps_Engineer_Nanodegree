""" Scoring
"""
import os
import pickle
import json
import pandas as pd
from sklearn.metrics import f1_score


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
test_data_filename = os.path.join(config['test_data_filename'])
model_path = os.path.join(config['output_model_path'])
trainedmodel_filename = os.path.join(config['trainedmodel_filename'])
score_filename = os.path.join(config['score_filename'])


def score_model():
    """Function for model scoring
    """
    # load test data
    test_data = pd.read_csv(os.path.join(test_data_path, test_data_filename))

    # load model
    with open(os.path.join(model_path, trainedmodel_filename), 'rb') as pickle_file:
        model = pickle.load(pickle_file)
    pickle_file.close()

    y_test = test_data['exited']
    x_test = test_data.drop(['corporation', 'exited'], axis=1)

    # model scoring
    y_pred = model.predict(x_test)
    f1 = f1_score(y_test, y_pred)
    print("f1 score = ", round(f1, 2))

    # save scoring
    with open(os.path.join(model_path, score_filename), 'w') as scoring_file:
        scoring_file.write("f1 score: " + str(round(f1, 2)))
    scoring_file.close()

    return f1


if __name__ == '__main__':
    score_model()
