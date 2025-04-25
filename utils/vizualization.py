import torch
import random
import numpy as np
import holoviews as hv
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns

hv.extension('bokeh')

pd.options.plotting.backend = 'holoviews'
hv.renderer('bokeh').theme = 'dark_minimal'

fix_seed = 2019
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

plt.style.use(['science', 'nature', 'no-latex'])
plt.rcParams['figure.dpi'] = 120
plt.rcParams.update({'font.size': 12})

COLORS = ["blue", "orange", "green", "red", "olivedrab"]
MODEL_NAMES = ['Intdisc_4']


def compute_time_lagged_mean_std(data, T, feature_index=0, end_idx=170):
    data = data[T:T + end_idx, :, :]
    length = data.shape[0] + data.shape[1]
    single_curve_dict = {n: [] for n in range(length)}

    for t in range(data.shape[0]):
        row = data[t, :, feature_index]
        for offset, idx in enumerate(range(t, t + len(row))):
            single_curve_dict[idx].append(row[offset])

    mean_curve = np.array([np.mean(single_curve_dict[idx]) for idx in single_curve_dict])
    std_curve = np.array([np.std(single_curve_dict[idx]) for idx in single_curve_dict])

    return mean_curve, std_curve


def plot_results_intervals_comp(results_dict, intervals, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 4))

    first_model_key = next(iter(results_dict))
    x_range = len(results_dict[first_model_key]["Mean"])
    x_vals = np.arange(x_range)

    for i, (l, u) in enumerate(intervals):
        model_key = f"({l}, {u})"
        curve_m = results_dict[model_key]["Mean"]
        curve_std = results_dict[model_key]["Std"]
        curve_t = results_dict[model_key]["True"]

        ax.fill_between(
            x_vals,
            curve_m - curve_std,
            curve_m + curve_std,
            color=COLORS[i],
            linewidth=1,
            linestyle='-',
            alpha=0.2
        )
        ax.plot(curve_t, color='black', linewidth=1, linestyle=':')
        ax.plot(curve_m, color=COLORS[i], linewidth=1, linestyle='-', label=model_key)
        ax.fill_between([0, x_range], [l, l], [u, u], color=COLORS[i], alpha=0.1)

    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()
    plt.savefig(save_path)


def sliding_plot_reg(results_dict, intervals, T=0, interval_length=24, step_size=6, save_path=True):
    sample_indices = np.arange(T, T + 24 * 2, step_size).astype(int)
    feature_index = 0
    spacing = 1
    fig, ax = plt.subplots(figsize=(4, 4))

    for j, interval in enumerate(intervals):
        preds, trues = results_dict[f'({interval[0]}, {interval[1]})']["Pred"], results_dict[f'({interval[0]}, {interval[1]})']["True"]

        true_color = '#1f1f1f'
        pred_color = COLORS[j]

        ax.tick_params(axis='both', which='major')
        ax.tick_params(axis='both', which='minor')

        for i, t in enumerate(sample_indices):
            pred_x = np.arange(t + 2 * interval_length, t + 3 * interval_length)
            y_min = interval[0] + spacing * i
            y_max = interval[1] + spacing * i

            ax.plot(np.arange(t, t + interval_length * 3),
                    np.array(list(trues[t, :, feature_index]) + \
                             list(trues[t + interval_length, :, feature_index]) + \
                             list(trues[t + 2 * interval_length, :, feature_index])) + spacing * i,
                    color=true_color, linewidth=0.5, label=f'True values' if j == 0 and i == 0 else "", zorder=5,
                    linestyle='--')
            ax.plot(np.arange(t + 2 * interval_length, t + 3 * interval_length),
                    preds[t + 2 * interval_length, :, feature_index] + spacing * i,
                    color=pred_color, alpha=0.9, linewidth=1,
                    label=f'Pred on ({interval[0]}, {interval[1]})' if i == 0 else "", zorder=6)
            ax.fill_between(pred_x, y_min, y_max, color=pred_color, alpha=0.1)

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()

    plt.savefig(save_path)


def sliding_plot_cls(results_dict, intervals, true_cls_index=3, T=0, interval_length=24, step_size=6, save_path=None):
    sample_indices = np.arange(T, T + 24 * 2, step_size).astype(int)
    feature_index = 0
    spacing = 1
    fig, ax = plt.subplots(figsize=(4, 4))

    for j, interval in enumerate(intervals):
        preds, trues = results_dict[f'({interval[0]}, {interval[1]})']["Pred"], results_dict[f'({interval[0]}, {interval[1]})']["True"]

        true_color = '#1f1f1f'
        pred_color = COLORS[j]

        ax.tick_params(axis='both', which='major')
        ax.tick_params(axis='both', which='minor')

        for i, t in enumerate(sample_indices):
            if true_cls_index in range(len(intervals)) and j == true_cls_index:
                ax.plot(np.arange(t, t + interval_length * 3),
                        np.array(list(trues[t, :, feature_index]) + \
                                 list(trues[t + interval_length, :, feature_index]) + \
                                 list(trues[t + 2 * interval_length, :, feature_index])) + spacing * i,
                        color=true_color, linewidth=0.5, label=f'True values' if i == 0 else "", zorder=5,
                        linestyle='--')

            ax.plot(np.arange(t + 2 * interval_length, t + 3 * interval_length),
                    preds[t + 2 * interval_length, :, feature_index] + spacing * i,
                    color=pred_color, alpha=0.9, linewidth=1,
                    label=f'Pred on ({interval[0]}, {interval[1]})' if i == 0 else "", zorder=6)

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()

    plt.savefig(save_path)
