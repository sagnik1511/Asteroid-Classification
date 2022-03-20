import os
from glob import glob


if __name__ == "__main__":
    data_dir = os.path.join("data", "raw")
    files = glob(f"{data_dir}/*json")
    for file in files:
        os.system(f"dvc add {file}")
    print("Files added in DVC...")
