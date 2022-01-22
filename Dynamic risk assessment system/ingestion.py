""" Ingestion

"""
import json
import glob
import pandas as pd


# Load config.json and get input and output paths
with open('config.json', 'r') as json_file:
    config = json.load(json_file)
json_file.close()

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
ingestedfile = config['ingestedfile']
finaldata_filename = config['finaldata_filename']


def merge_multiple_dataframe():
    """ Function for data ingestion
    """
    # check for datasets, compile them together, and write to an output file

    # read all files from data folder.
    files = glob.glob(input_folder_path + "/*.csv")

    # compile the different dataframes into single dataframe
    dfs = []
    for file in files:
        dfs.append(pd.read_csv(file, index_col=None))
    data = pd.concat(dfs, axis=0, ignore_index=True)

    # drop duplicates
    data.drop_duplicates(inplace=True)

    # write df to the workspace
    data.to_csv(finaldata_filename, index=False)

    # save a record of all files
    with open(output_folder_path + "/" + ingestedfile, "w") as record:
        for file in files:
            record.write(file.split("/")[1] + "\n")
    record.close()


if __name__ == '__main__':
    merge_multiple_dataframe()
