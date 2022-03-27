import os
import sys
import yaml
import argparse
from pathlib import Path
import pandas as pd
sys.path.append("../asteroid_classification/src")
from utils.utils import *


def preprocessing(train_dataframe, test_dataframe, job_config):
    # creating a copy of both dataset
    print("Creating copy of both dataset ...")
    train_data, test_data = train_dataframe.copy(), test_dataframe.copy()

    # performing attribute_extraction
    print("Performing attribute extraction ...")
    train_data = attribute_extraction(train_data)
    test_data = attribute_extraction(test_data)

    # processing time feature
    print("Processing datetime features")
    train_data = process_orb_det_time(train_data)
    test_data = process_orb_det_time(test_data)
    train_data = process_capture_date(train_data)
    test_data = process_capture_date(test_data)

    # omitting unused / unwanted features
    print("Filtering required columns")
    train_data.drop(job_config["unused_cols"], axis=1, inplace=True)
    test_data.drop(job_config["unused_cols"], axis=1, inplace=True)

    # encoding categorical columns
    print("Encoding ...")
    for fet in job_config["obj_cols"]:
        train_data[fet], test_data[fet] = encode(train_data[fet], test_data[fet])

    # defining columns which need to be ignored when performing typecasting
    ignore_cols = job_config["obj_cols"] + job_config["date_time_cols"]

    # performing typecasting
    for fet in train_data.columns:
        if fet not in ignore_cols:
            train_data[fet] = train_data[fet].astype("float")
            test_data[fet] = test_data[fet].astype("float")

    # performing standard scaling
    print("Performing scaling ...")
    for fet in job_config["scaling"]:
        train_data, test_data = scaling(train_data, test_data, fet)

    return train_data, test_data


def process_data(conf_path):
    # validating if the config is present or not
    if not os.path.isfile(conf_path):
        print("Config file not found!!!")
        return

    # loading config data
    job_config = yaml.safe_load(open(conf_path, "rb"))
    print("configurations from {} loaded...".format(conf_path))

    # fetching input data paths
    train_data_base_path = job_config["input_path"]["train_set"]
    test_data_base_path = job_config["input_path"]["test_set"]

    # validating input data paths
    assert os.path.isfile(train_data_base_path), "train set not found!!!"
    assert os.path.isfile(test_data_base_path), "test set not found!!!"

    # loading input data
    print("Loading data ...")
    train_df = pd.read_csv(train_data_base_path)
    test_df = pd.read_csv(test_data_base_path)

    # passing the data for processing
    train_df, test_df = preprocessing(train_df, test_df, job_config)

    # fetching the output data paths
    train_op_path = job_config["output_path"]["train_set"]
    test_op_path = job_config["output_path"]["test_set"]

    # storing the outputs
    print("Storing processed data ...")
    train_df.to_csv(train_op_path, index=False)
    test_df.to_csv(test_op_path, index=False)


if __name__ == "__main__":
    print(os.getcwd())
    # assigning parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--conf", type=Path, required=True, metavar="", help="config path")
    args = parser.parse_args()

    # calling preprocessing
    process_data(args.conf)

    print("Process completed successfully...")
