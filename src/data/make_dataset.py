import os
import yaml
import argparse
from glob import glob
from pathlib import Path
from src.utils.utils import create_dataframe


SEED = 42
DEFAULT_OUTPUT_DIRECTORY = os.path.join("data", "primary")


def prepare_dataset(conf_path, op_dir):

    if not os.path.isfile(conf_path):
        print("Config file not found!!!")
        return

    # checking whether output directory is present or not
    if not os.path.isdir(op_dir):
        print("output directory not found. Generating output directory!!!")
        os.mkdir(op_dir)

    # loading config data
    conf_data = yaml.safe_load(open(conf_path, "rb"))
    print("configurations from {} loaded...".format(conf_path))

    # fetching record paths
    record_paths = glob(f"{conf_data['root_dir']}/*json")
    train_count = int(len(record_paths) * conf_data["data_split"]["train_size"])
    train_records = record_paths[:train_count]
    test_records = record_paths[train_count:]
    print("Creating primary datasets...")

    # processing raw data
    training_set = create_dataframe(train_records)
    testing_set = create_dataframe(test_records)

    # saving processed data
    train_path = os.path.join(op_dir, "train.csv")
    test_path = os.path.join(op_dir, "test.csv")
    print(f"Training set stored at {train_path}")
    print(f"Testing set stored at {test_path}")
    training_set.to_csv(train_path, index=False)
    testing_set.to_csv(test_path, index=False)

    print("Process completed...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--conf", type=Path, required=True, metavar="", help="config path")
    parser.add_argument("-d", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY,
                        metavar="", help="output file directory")

    args = parser.parse_args()
    prepare_dataset(args.conf, args.output_dir)
