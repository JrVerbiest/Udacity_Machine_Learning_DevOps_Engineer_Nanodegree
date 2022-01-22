"""Deployment script

"""
import os
import json
import shutil

# Load config.json and correct path variable
with open('config.json', 'r') as json_file:
    config = json.load(json_file)
json_file.close()

model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
trainedmodel_filename = os.path.join(config['trainedmodel_filename'])
score_filename = os.path.join(config['score_filename'])
output_folder_path = os.path.join(config['output_folder_path'])
ingestedfile = os.path.join(config['ingestedfile'])


def store_model_into_pickle():
    """function for deployment
    """
    shutil.copy2(
        os.path.join(
            model_path,
            trainedmodel_filename),
        prod_deployment_path)
    shutil.copy2(
        os.path.join(
            model_path,
            score_filename),
        prod_deployment_path)
    shutil.copy2(
        os.path.join(
            output_folder_path,
            ingestedfile),
        prod_deployment_path)


if __name__ == '__main__':
    store_model_into_pickle()
