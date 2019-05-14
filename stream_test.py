import pandas as pd
import time

start = time.time();

# steam all rows by "chunksize" rows at a time
for chunk in pd.read_csv("data/raw/ml_videos.csv", chunksize=1000):
    for index, row in chunk.iterrows():
        print(row)

# read rows from (skiprows) to (skiprows+nrows)
print(pd.read_csv("data/raw/ml_videos.csv", nrows=10, skiprows=5))

print(time.time()-start)