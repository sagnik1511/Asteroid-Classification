import pandas as pd

data_root = "data/interim/"

for path in ["train", "test"]:
    full_path = data_root + path + ".csv"
    df = pd.read_csv(full_path)
    df = df.sample(frac=0.1)
    df.to_csv(f"{data_root}{path}_small.csv")
