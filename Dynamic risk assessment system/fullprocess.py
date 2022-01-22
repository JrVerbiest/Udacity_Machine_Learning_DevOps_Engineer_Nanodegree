"""fullprocess
"""
import os
import json
import sys
from sklearn.metrics import f1_score

from diagnostics import model_predictions
from training import train_model
from deployment import store_model_into_pickle


# Load config.json and get environment variables
with open('config.json', 'r') as json_file:
    config = json.load(json_file)
json_file.close()

def main():

    with open(os.path.join(os.path.join(config['prod_deployment_path']), "ingestedfiles.txt")) as ingested_file:
        ingested_files = ingested_file.read().splitlines()[-1]
    ingested_file.close()

    dataset_files = os.listdir(os.path.join(config['input_folder_path']))

    # exit if there no new datasets
    if (ingested_files == dataset_files):
        sys.exit() 
    
    with open(os.path.join(os.path.join(config['prod_deployment_path']), "latestscore.txt")) as score_file:
        #we only read one line -> dataset2.csv is our new dataset
        lastest_score = float(score_file.read().splitlines()[0].split(":")[1])
    ingested_file.close()
    
    new_dataset = [file for file in dataset_files if not file in ingested_files][0] 

    input_folder_path = os.path.join(config['input_folder_path'])
    model_pth = os.path.join(config['prod_deployment_path'])
    model_file = os.path.join(config['trainedmodel_filename'])
    model, model_predict, y, _ = model_predictions(input_folder_path, new_dataset, model_pth, model_file)
    new_score = round(f1_score(y, model_predict),2)


    # exit if lastest_score smaller or equal
    if (lastest_score >= new_score):
        sys.exit() 
        
    train_model()
    store_model_into_pickle()

    os.system("python reporting.py")
    os.system("python apicalls.py")


if __name__ == '__main__':
    main()
