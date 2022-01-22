"""API Calls
"""
import os
import json
import time
import requests


URL = "http://127.0.0.1:8000"


headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

response1 = requests.post(
    "%s/prediction" %
    URL,
    json={
        "dataset_path": "testdata.csv"},
    headers=headers).text
response2 = requests.get("%s/scoring" % URL, headers=headers).text
response3 = requests.get("%s/summarystats" % URL, headers=headers).text
response4 = requests.get("%s/diagnostics" % URL, headers=headers).text

report = response1 + "\n\n" + response2 + \
    "\n\n" + response3 + "\n\n" + response4

with open('config.json', 'r') as json_file:
    config = json.load(json_file)
json_file.close()

model_path = os.path.join(config['output_model_path'])

with open(os.path.join(model_path, f'apireturns_{time.strftime("%y%m%d%H%M%S")}.txt'), "w") as apireturn_file:
    apireturn_file.write(report)
apireturn_file.close()
