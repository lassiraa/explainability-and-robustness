import json
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

path = 'data/'
files = os.listdir(path)
results = []

for file in files:
    if 'stability' not in file:
        continue
    with open(f'{path}{file}', 'r') as f:
        meta = file.split('_')
        model_name = meta[0]
        method = meta[-2]
        res = np.array(json.load(f))
        res = res[~np.isnan(res)]
        mean = res.mean()
        var = res.var()
        results.append(dict(
            model=model_name,
            method=method,
            mean=mean,
            var=var
        ))

results_df = pd.DataFrame.from_records(results)
results_df.sort_values(by=['method', 'model'], inplace=True)
print(results_df)