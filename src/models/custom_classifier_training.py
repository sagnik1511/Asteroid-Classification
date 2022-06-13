import os
import sys
import yaml
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
sys.path.append("src")
from models.model_warehouse import warehouse_dict
from utils.utils import store_artifact, omit_outliers, save_data
from models.metrics import make_training_report


def training(model_config, train, val, test, testing=False):
    # fetching tuning parameters
    params = model_config["params"]

    # unpacking training and validation data
    x_train, y_train = train
    x_val, y_val = val
    x_test, y_test = test

    # loading GridSearchCV model
    print("Preparing base model ...")
    base_model_name = model_config["model_name"]
    base_model = warehouse_dict[base_model_name]()
    gcv_model = GridSearchCV(base_model, params)

    # training model
    print("Training model ...")
    gcv_model.fit(x_train, y_train)
    print("Training finished ...")

    # storing retrained best estimator and training metrics
    print("Finding best estimator and retraining for artifact storing ...")
    best_params = gcv_model.best_params_
    best_estimator = gcv_model.best_estimator_
    print(f"best estimator found : {best_estimator}")
    best_estimator.fit(x_train, y_train)
    if not testing:
        report_output_directory = os.path.join(model_config["output_artifact_report_dir"], base_model_name)
    else:
        report_output_directory = os.path.join("reports/figures", "training_report")
    if not os.path.isdir(report_output_directory):
        os.mkdir(report_output_directory)
    make_training_report(gcv_model, x_train, y_train,
                         x_val, y_val, x_test, y_test,
                         report_output_directory,
                         best_params, connect_mlflow_logger=testing)
    artifact_output_directory = os.path.join(model_config["output_artifact_dir"], base_model_name)
    if not os.path.isdir(artifact_output_directory):
        os.mkdir(artifact_output_directory)
    if not testing:
        roc_curve_dir = os.path.join(model_config["figures"]["roc_curve"])
    else:
        roc_curve_dir = os.path.join("reports/figures", "training_report")
    if not testing:
        store_artifact(best_estimator, artifact_output_directory)



def load_and_hot_process(process_config):

    # fetching input data paths
    train_data_base_path = process_config["input_path"]["train_set"]
    test_data_base_path = process_config["input_path"]["test_set"]

    # validating input data paths
    assert os.path.isfile(train_data_base_path), "train set not found!!!"
    assert os.path.isfile(test_data_base_path), "test set not found!!!"

    # loading input data
    print("Loading Data ...")
    train_df = pd.read_csv(train_data_base_path)
    test_df = pd.read_csv(test_data_base_path)

    # partitioning feature set and target variable
    print("splitting data into independent and dependent feature set")
    x_train, y_train = train_df.iloc[:, :-1], train_df.iloc[:, -1]
    x_test, y_test = test_df.iloc[:, :-1], test_df.iloc[:, -1]

    # applying variance threshold
    print("Applying variance threshold ...")
    var_thresh = process_config["preprocess"]["variance_threshold"]
    var_job = VarianceThreshold(var_thresh)
    train_vector = var_job.fit_transform(x_train)
    test_vector = var_job.transform(x_test)

    # splitting train-validation
    print("Splitting data into train and validation set ...")
    x_train, x_val, y_train, y_val = train_test_split(train_vector, y_train,
                                                      test_size=process_config["train_val_split"])

    return (x_train, y_train), (x_val, y_val), (test_vector, y_test)


def classifier_training(process_config_path, model_config_path, testing=False):
    # validating if the config is present or not
    if not os.path.isfile(process_config_path):
        print("Processing config file not found!!!")
        return
    if not os.path.isfile(model_config_path):
        print("Model config file not found!!!")
        return

    # loading config data
    process_config = yaml.safe_load(open(process_config_path, "rb"))
    print("configurations from {} loaded...".format(process_config_path))
    model_config = yaml.safe_load(open(model_config_path, "rb"))
    print("configurations from {} loaded...".format(model_config_path))

    # loading and processing data for training
    train, val, test = load_and_hot_process(process_config)

    # training and storing all necessary details
    training(model_config, train, val, test, testing=testing)

    if not testing:
        # storing processed data
        train_op_path = process_config["output_path"]["train_set"]
        val_op_path = process_config["output_path"]["val_set"]
        test_op_path = process_config["output_path"]["test_set"]

        save_data(train, train_op_path)
        save_data(val, val_op_path)
        save_data(test, test_op_path)


if __name__ == "__main__":
    # assigning parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-pc", "--proc_conf", type=Path, required=True, metavar="", help="config path for processing")
    parser.add_argument("-mc", "--model_conf", type=Path, required=True, metavar="", help="config path for model")
    args = parser.parse_args()

    # training classifier
    classifier_training(args.proc_conf, args.model_conf)

    print("Process completed successfully...")
