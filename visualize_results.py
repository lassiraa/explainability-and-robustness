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
    if 'results' not in file:
        continue
    model_name = file.replace('_results.json', '')
    with open(f'{path}{file}', 'r') as f:
        model_res = json.load(f)
        results += model_res

results_df = pd.DataFrame.from_records(results)
results_df = results_df[results_df['distortion_method'] == 'singular_spike']
sns.set_theme(style="ticks", color_codes=True)
ax = sns.scatterplot(x='distort_background', y='distort_ratio', hue='model_name', style='model_name', data=results_df)
plt.show()
