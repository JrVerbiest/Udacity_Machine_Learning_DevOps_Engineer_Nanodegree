"""API Setup
"""

import json
import os
from flask import Flask, request
from diagnostics import model_predictions, dataframe_summary, dataframe_missing_data, execution_time, outdated_packages_list
from scoring import score_model


# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'


@app.route('/')
def index():
    """Small test
    """
    user = request.args.get("user")
    return "Hello " + user + "\n"


@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    """Prediction Endpoint
    """

    with open('config.json', 'r') as json_file:
        config = json.load(json_file)
    json_file.close()

    data_file = request.get_json()['dataset_path']

    model_pth = os.path.join(config['prod_deployment_path'])
    model_file = os.path.join(config['trainedmodel_filename'])
    test_data_path = os.path.join(config['test_data_path'])

    _, model_predict, _, _ = model_predictions(
        test_data_path, data_file, model_pth, model_file)

    return str(model_predict)


@app.route("/scoring", methods=['GET', 'OPTIONS'])
def scoring():
    """Scoring Endpoint
    """

    score = score_model()
    return str(score)


@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats():
    """Summary Statistics Endpoint
    """
    statistics = dataframe_summary()
    return str(statistics)


@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnostics():
    """Diagnostics Endpoint
    """

    with open('config.json', 'r') as json_file:
        config = json.load(json_file)
    json_file.close()

    data_file = os.path.join(config['finaldata_filename'])
    missing_data = str(dataframe_missing_data(data_file))
    exec_time = str(execution_time())
    outd_packages_list = str(outdated_packages_list())

    return missing_data + "\n" + exec_time + "\n" + outd_packages_list


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
