import json
import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.preprocessing import LabelEncoder, StandardScaler


MASTER_DF_COLUMNS = ["id", "date", "name", "absolute_magnitude", "diameter",
                     "relative_velocity", "miss_distance", "orbital_data", "hazardous"]


def fetch_single_record(data, index, master_index):
    date = str(list(data["near_earth_objects"].keys())[0])
    inner_records = data["near_earth_objects"][date][index]
    ref_id = str(inner_records["neo_reference_id"])
    name = str(inner_records["name"])
    abs_magnitude = str(inner_records["absolute_magnitude_h"])
    diameter = str(inner_records["estimated_diameter"])
    rel_velocity = str(inner_records["close_approach_data"][0]["relative_velocity"])
    m_distance = str(inner_records["close_approach_data"][0]["miss_distance"])
    orb_data = str(inner_records["orbital_data"])
    is_hazardous = str(inner_records["is_potentially_hazardous_asteroid"])

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
            "hazardous": is_hazardous,
        }, index=[master_index]
    )

    return single_record


def create_dataframe(json_file_list):

    # defining master index
    master_index = 0

    # creating a base dataframe object with selective columns
    op_df = pd.DataFrame(columns=MASTER_DF_COLUMNS)

    # iterating through json files
    for json_file_path in json_file_list:
        print(f"File taken : {json_file_path}")
        # loading json file
        json_data = json.load(open(json_file_path, "rb"))
        try:
            # fetching number of records from individual file
            num_el = json_data["element_count"]
            # iterating through the records of the file
            for index in range(num_el):
                # fetching single records
                record = fetch_single_record(json_data, index, master_index)

                # concatenating with base dataframe
                op_df = pd.concat([op_df, record], axis=0)
                master_index += 1
        except:
            pass

    return op_df


def diameter_processing(data):
    # creating a copy of the dataset
    temp_df = data.copy()

    # fetching the only needed values
    temp_df["max_diameter"] = \
        temp_df["diameter"].apply(lambda x: literal_eval(x)["kilometers"]["estimated_diameter_max"])
    temp_df["min_diameter"] = \
        temp_df["diameter"].apply(lambda x: literal_eval(x)["kilometers"]["estimated_diameter_min"])

    return temp_df


def process_orb_det_time(dataframe):
    # creating a copy of the dataset
    df = dataframe.copy()

    # splitting the feature values so that we can get segmented features
    df[["orb_date", "orb_time"]] = df["orbit_determination_date"].str.split(" ", expand=True)
    df[["orb_year", "orb_month", "orb_day"]] = df["orb_date"].str.split("-", expand=True)
    try:
        df[["orb_hour", "orb_min", "orb_sec"]] = df["orb_time"].str.split("-", expand=True)
    except:
        pass

    return df


def process_capture_date(dataframe):
    # creating a copy of the dataset
    df = dataframe.copy()

    # splitting the feature values so that we can get segmented features
    df[["cap_year", "cap_month", "cap_day"]] = df["date"].str.split("-", expand=True)

    return df


def dict_to_pd(data):
    # loading the string data into dict format
    data = literal_eval(data)

    # fetching only the values from the dict object
    val_str = ",".join([v for _, v in data.items()])

    return val_str


def encode(train_feature, test_feature):
    # defining encoder
    scaler = LabelEncoder()

    # fitting on train data and updating on both dataset
    train_feature = scaler.fit_transform(train_feature)
    test_feature = scaler.transform(test_feature)

    return train_feature, test_feature


def scaling(train_dataframe, test_dataframe, feature):
    # creating a copy of both dataset
    train_df, test_df = train_dataframe.copy(), test_dataframe.copy()

    # defining standard scaler
    scaler = StandardScaler()

    # fitting on train data and updating on both dataset
    train_df[feature] = scaler.fit_transform(train_df[[feature]])
    test_df[feature] = scaler.transform(test_df[[feature]])

    return train_df, test_df


def remove_dispersed_and_condensed_features(train_dataframe, test_dataframe,
                                            up_thresh=0.95, down_thresh=0.05):
    # creating a copy of both dataset
    train_df = train_dataframe.copy()
    test_df = test_dataframe.copy()

    # iterating through the features
    for fet in test_df.columns:
        var = np.var(train_df[fet])

        # checking if the feature is is the variance bound
        if up_thresh > var or down_thresh < var:
            print(f"{fet} removed for bad distribution...")

            # if not then deleting them
            train_df.drop(fet, 1, inplace=True)
            test_df.drop(fet, 1, inplace=True)

    return train_df, test_df


def process_orb_data(data):
    # fetching the keys of the dict data stored in string format
    cols = list(literal_eval(data["orbital_data"][0]).keys())

    # applying required function
    temp_df = data["orbital_data"].apply(lambda x: dict_to_pd(x))

    # typecasting the numpy string to pandas.DataFrame
    temp_df = pd.DataFrame(temp_df)

    # creating a new dataframe
    op_df = pd.DataFrame()

    # expanding the values to generate several features
    op_df[cols] = temp_df["orbital_data"].str.split(",", expand=True)

    return op_df


def obj2float(data, ignore_cols):
    # creating a copy of the dataset
    temp_df = data.copy()

    # iterating through features
    for col in temp_df.columns:
        # checking if the feature is categorical and also not inside the specific columns
        if temp_df[col].dtype == "object" and col not in ignore_cols:

            # performing typecasting
            temp_df[col] = temp_df[col].astype("float")

    return temp_df


def attribute_extraction(dataframe):
    # creating a copy of the dataset
    data = dataframe.copy()

    # performing processing functions
    data = diameter_processing(data)

    # extracting attributes
    data["relative_velocity"] = data["relative_velocity"].apply(lambda x: literal_eval(x)["kilometers_per_second"])
    data["miss_distance"] = data["miss_distance"].apply(lambda x: literal_eval(x)["kilometers"])

    # concatenating features with processed orbital data
    data = pd.concat([data, process_orb_data(data)], axis=1)

    return data
