""" Training

"""
import pickle
import os
import json
import pandas as pd
from sklearn.linear_model import LogisticRegression


# Load config.json and get path variables
with open('config.json', 'r') as json_file:
    config = json.load(json_file)
json_file.close()

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])
finaldata_filename = config['finaldata_filename']
trainedmodel_filename = config['trainedmodel_filename']


def train_model():
    """ Function for training the model
    """
    # load data
    data = pd.read_csv(os.path.join(finaldata_filename))

    # use this logistic regression for training
    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class='auto',
        n_jobs=None,
        penalty='l2',
        random_state=0,
        solver='liblinear',
        tol=0.0001,
        verbose=0,
        warm_start=False)

    y_train = data["exited"]
    x_train = data.drop(["corporation", "exited"], axis=1)
    # fit the logistic regression to your data
    model.fit(x_train, y_train)

    # write the trained model to your workspace in a file called
    # trainedmodel.pkl
    with open(os.path.join(model_path, trainedmodel_filename), 'wb') as pickle_file:
        pickle.dump(model, pickle_file)
    pickle_file.close()


if __name__ == '__main__':
    train_model()
