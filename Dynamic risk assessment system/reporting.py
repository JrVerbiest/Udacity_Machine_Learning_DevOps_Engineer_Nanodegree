"""Reporting
"""
import json
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from diagnostics import model_predictions

# Load config.json and get path variables
with open('config.json', 'r') as json_file:
    config = json.load(json_file)
json_file.close()

data_path = os.path.join(config['test_data_path'])
data_filename = os.path.join(config['test_data_filename'])
model_path = os.path.join(config['output_model_path'])
model_filename = os.path.join(config['trainedmodel_filename'])
confusionmatrix = os.path.join(config['confusionmatrix'])


def score_model():
    """Function for reporting
    """
    model, _, y, X = model_predictions(
        data_path, data_filename, model_path, model_filename)
    plot_confusion_matrix(model, X, y)
    filename_confusionmatrix = confusionmatrix.split(".")[0]+"_"+str(time.strftime("%y%m%d%H%M%S"))+".png"
    plt.savefig(os.path.join(model_path, filename_confusionmatrix))
    

if __name__ == '__main__':
    score_model()
