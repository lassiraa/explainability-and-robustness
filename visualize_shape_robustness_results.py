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
    if 'shape_robustness' not in file:
        continue
    model_name = file.replace('_shape_robustness.json', '')
    with open(f'{path}{file}', 'r') as f:
        model_res = json.load(f)
        results += model_res

results_df = pd.DataFrame.from_records(results)
results_no_distort = results_df[results_df['distort_background'].isnull()]
results_blur = results_df[results_df['distort_background'] == 'blur']
sns.set_theme(style="ticks", color_codes=True)
fig, axs = plt.subplots(ncols=2)
sns.scatterplot(x='distortion_method', y='distort_ratio', hue='model_name',
                style='model_name', data=results_no_distort, ax=axs[0], s=80)
axs[0].set_title('Recovered accuracy with no background alteration')
sns.scatterplot(x='distortion_method', y='distort_ratio', hue='model_name',
                style='model_name', data=results_blur, ax=axs[1], s=80)
axs[1].set_title('Recovered accuracy with blurred background')
plt.show()
