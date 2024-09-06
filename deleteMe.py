import pandas as pd

df = pd.read_csv("2024-08-30_22_27_15_test_dataset_drone.csv")
print(df)
print(df.columns)
print(df['filename'])