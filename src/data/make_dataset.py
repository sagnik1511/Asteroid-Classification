import os
import json
import yaml
import logging
import argparse
import pandas as pd
from glob import glob
from pathlib import Path


SEED = 42
DEFAULT_OUTPUT_DIRECTORY = os.path.join("data", "primary")
MASTER_DF_COLUMNS = ["id", "date", "name", "absolute_magnitude", "diameter",
                     "relative_velocity", "miss_distance", "orbital_data"]


def fetch_single_record(data, index, master_index):
    date = list(data["near_earth_objects"].keys())[0]
    inner_records = data["near_earth_objects"][date][index]
    ref_id = inner_records["neo_reference_id"]
    name = inner_records["name"]
    abs_magnitude = inner_records["absolute_magnitude_h"]
    diameter = str(inner_records["estimated_diameter"])
    rel_velocity = str(inner_records["close_approach_data"][0]["relative_velocity"])
    m_distance = str(inner_records["close_approach_data"][0]["miss_distance"])
    orb_data = str(inner_records["orbital_data"])

    single_record = pd.DataFrame(
        {
            "id": ref_id,
            "date": date,
            "name": name,
            "absolute_magnitude": abs_magnitude,
            "diameter": diameter,
            "relative_velocity": rel_velocity,
            "miss_distance": m_distance,
            "orbital_data": orb_data,
        }, index=[master_index]
    )

    return single_record


def create_dataframe(json_file_list):
    master_index = 0
    op_df = pd.DataFrame(columns=MASTER_DF_COLUMNS)
    for json_file_path in json_file_list:
        print(f"File taken : {json_file_path}")
        json_data = json.load(open(json_file_path, "rb"))
        try:
            num_el = json_data["element_count"]
            for index in range(num_el):
                record = fetch_single_record(json_data, index, master_index)
                op_df = pd.concat([op_df, record], axis=0)
                master_index += 1
        except:
            pass

    return op_df


def prepare_dataset(conf_path, op_dir):

    # setting logger
    logger = logging.getLogger(__name__)
    if not os.path.isfile(conf_path):
        logger.error("Config file not found!!!")
        return

    # checking whether output directory is present or not
    if not os.path.isdir(op_dir):
        logger.info("output directory not found. Generating output directory!!!")
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
