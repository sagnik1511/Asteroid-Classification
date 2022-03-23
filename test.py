from sklearn.feature_selection import VarianceThreshold
import pandas as pd

df = pd.read_csv("data/primary/test.csv", usecols = ["id", "absolute_magnitude"])
scaler = VarianceThreshold(0.9)

op = scaler.fit_transform(df)
print(op)