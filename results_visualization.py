import json
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def read_shape_robustness(path='./data/'):
    files = os.listdir(path)
    results = []

    for file in files:
        if 'shape_robustness' not in file:
            continue
        with open(f'{path}{file}', 'r') as f:
            model_res = json.load(f)
            results += model_res
    
    return pd.DataFrame.from_records(results)


def read_stability(path='./data/'):
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
                model_name=model_name,
                method=method,
                mean=mean,
                var=var
            ))
    
    return pd.DataFrame.from_records(results).sort_values(by=['method', 'model_name'])


def read_weighting_game(path='./data/'):
    files = os.listdir(path)
    results = []

    for file in files:
        if 'weighting_game' not in file:
            continue
        with open(f'{path}{file}', 'r') as f:
            meta = file.split('-')
            model_name = meta[0]
            method = meta[1]
            res = json.load(f)
            accuracies = np.array([entry['accuracy'] for entry in res])
            areas = np.array([entry['object_area'] for entry in res])
            areas = areas[~np.isnan(accuracies)]
            accuracies = accuracies[~np.isnan(accuracies)]
            # bins = np.logspace(np.log10(areas.min()), np.log10(areas.max()), 10)
            num_bins = 40
            bins = np.linspace(np.floor(areas.min()), np.ceil(areas.max()), num_bins)
            digitized = np.digitize(areas, bins) / num_bins
            mean = np.mean(accuracies)
            results.append(dict(
                model_name=model_name,
                method=method,
                mean=mean,
                accuracies=accuracies,
                digitized=digitized
            ))
    
    return pd.DataFrame.from_records(results).sort_values(by=['method', 'model_name'])


def clean_model_name(data):
    data['model_name'] = data['model_name'].replace(dict(
        vit_b_32='ViT-B/32',
        vit='ViT-B/32',
        resnet50='ResNet50',
        vgg16_bn='VGG16-BN',
        vgg16='VGG16-BN',
        swin_t='Swin-T',
        swin='Swin-T'
    ))
    return data
    


def plot_shape_robustness(data):
    data = clean_model_name(data)
    data['distort_background'].fillna('None', inplace=True)
    data['distort_background'].replace(dict(blur='Blur', remove='Remove'), inplace=True)
    sns.set_theme(style='ticks', color_codes=True)
    ax = sns.scatterplot(
        x='distort_background',
        y='distort_ratio',
        hue='model_name',
        style='model_name',
        data=data,
        s=80
    )
    ax.set(xlabel='Background processing', ylabel='Retained accuracy')
    ax.legend(title='Model')
    plt.show()


def plot_stability(data, by='method', labels='model_name'):
    data = clean_model_name(data)
    sns.set_theme(style='ticks', color_codes=True)
    ax = sns.barplot(x=by, y='mean', hue=labels, data=data)
    ax.set(xlabel='Method', ylabel='Mean correlation')
    ax.legend(title='Model')
    plt.show()


def plot_weighting_game(data, by='method', labels='model_name'):
    data = clean_model_name(data)
    sns.set_theme(style='ticks', color_codes=True)
    ax = sns.scatterplot(
        x=by,
        y='mean',
        hue=labels,
        style=labels,
        data=data,
        s=80
    )
    ax.set(xlabel='Method', ylabel='Mean accuracy')
    ax.legend(title='Model')
    plt.show()


def plot_binned_weighting_game(data):
    data = clean_model_name(data)
    indices = [(0,0), (0,1), (1,0), (1,1)]
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i, model_name in enumerate(data['model_name'].unique()):
        pruned = data[data['model_name'] == model_name]
        exploded = pruned.explode(['accuracies', 'digitized'])
        grouped = exploded.groupby(['method', 'digitized'])\
            .agg({'accuracies': 'mean'})\
            .reset_index()
        ax = sns.lineplot(
            x='digitized',
            y='accuracies',
            hue='method',
            style='method',
            data=grouped,
            ax=axs[indices[i]]
        )
        ax.get_legend().remove()
        ax.set(title=model_name, ylabel=None, xlabel=None)
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines[:6], labels[:6], loc='upper center')
    fig.supxlabel('Ratio of object area to size of image')
    fig.supylabel('Mean accuracy')
    plt.show()


if __name__ == '__main__':
    data = read_weighting_game()
    plot_binned_weighting_game(data)