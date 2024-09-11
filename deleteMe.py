from helper import *    
import seaborn as sns
import os
import struct
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import ast



depthDf = pd.read_csv("depth.csv")

print(depthDf.columns)
for index,row in depthDf.iterrows():
    i=0
    li = ast.literal_eval(row["x"])
    print(li[0])
    break
# print(type(depthDf["x"][0]))