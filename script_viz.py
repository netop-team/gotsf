import json
import ast
import pickle
import subprocess
import torch
import random
import numpy as np
import holoviews as hv
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns
from io import StringIO
import os
import argparse
from utils.tools import separate_interval

# Set up visualization backend and plot options
hv.extension('bokeh')
pd.options.plotting.backend = 'holoviews'
hv.renderer('bokeh').theme = 'dark_minimal'

# Seed fixing for reproducibility
fix_seed = 2019
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# Matplotlib configuration
plt.style.use(['science', 'nature', 'no-latex'])
plt.rcParams['figure.dpi'] = 240
plt.rcParams['savefig.dpi'] = 240
plt.rcParams.update({'font.size': 12})
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

# Color palette for plots
COLORS = ["blue", "orange", "green", "red", "purple", "brown", "olive", "cyan", "pink", "yellow"]


class ModelVisualization:
    def __init__(self, args):
        self.args = args

        if os.path.isdir(os.path.join(self.args.checkpoints_dir, self.args.experiment_name)):
            self.model_dir = os.path.join(self.args.checkpoints_dir, self.args.experiment_name)
        else:
            raise ValueError('Experiment directory does not exist')

        self.train_loss = np.load(self.model_dir + '/train_loss.npy')
        self.model_args = pickle.load(open(self.model_dir + '/args.pk', 'rb'))
        if self.model_args.training_interval_technique in ['interval-uniform', 'interval-discrete']:
            train_int_min, train_int_max = self.model_args.train_interval_l, self.model_args.train_interval_u
            self.train_discrete_intervals = separate_interval(
                interval=[train_int_min, train_int_max],
                nr_intervals=self.args.nr_intervals,
            )
            self.eval_train_discrete_intervals = separate_interval(
                interval=[train_int_min, train_int_max],
                nr_intervals=self.model_args.nr_intervals,
            )
        if self.model_args.training_interval_technique == 'interval-discrete':
            self.results_dict_discrete = self.build_results_dict_discrete()
        elif self.model_args.training_interval_technique == 'interval-uniform':
            self.results_dict_uniform = self.build_results_dict_uniform()
        else:
            self.results_dict_normal = self.build_results_dict_normal()

        self.save_path = self.args.plots_dir + '/' + self.args.experiment_name + '/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def load_data_combined(self, interval):
        results_path = self.args.results_dir + '/' + self.args.experiment_name + '/' + f'({interval[0]}, {interval[1]})/'
        preds = np.load(os.path.join(results_path, 'pred.npy'))
        trues = np.load(os.path.join(results_path, 'true.npy'))
        return preds, trues

    def load_data_discrete(self, interval):
        results_path = self.args.results_dir + '/' + self.args.experiment_name + '/' + f'({interval[0]}, {interval[1]})/'
        preds = np.load(os.path.join(results_path, 'pred.npy'))
        trues = np.load(os.path.join(results_path, 'true.npy'))
        preds_cls = np.load(os.path.join(results_path, 'pred_cls.npy'))
        trues_cls = np.load(os.path.join(results_path, 'true_cls.npy'))
        return preds, trues, preds_cls, trues_cls

    def build_results_dict_discrete(self):
        results = dict()
        for interval in self.train_discrete_intervals + self.eval_train_discrete_intervals:
            if interval in self.eval_train_discrete_intervals:
                if not os.path.exists(self.args.results_dir + '/' + self.args.experiment_name + '/'):
                    os.makedirs(os.path.dirname(self.args.results_dir + '/' + self.args.experiment_name + '/'))
                if f'({interval[0]}, {interval[1]})' not in os.listdir(
                        self.args.results_dir + '/' + self.args.experiment_name + '/'):
                    subprocess.run(
                        ["bash", "./scripts/multivariate_forecasting/Wireless/stats.sh", str(interval[0]),
                         str(interval[1])],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                preds, trues, preds_cls, trues_cls = self.load_data_discrete(interval)

                reg_preds_mean, reg_preds_std = self.compute_time_lagged_mean_std(preds)
                reg_trues_mean, _ = self.compute_time_lagged_mean_std(trues)

                cls_preds_mean, cls_preds_std = self.compute_time_lagged_mean_std(preds_cls)
                cls_trues_mean, _ = self.compute_time_lagged_mean_std(trues_cls)

                results[f'({interval[0]}, {interval[1]})'] = {
                    "Reg_Mean": reg_preds_mean,
                    "Reg_Std": reg_preds_std,
                    "Reg_True": reg_trues_mean,
                    "Cls_Mean": cls_preds_mean,
                    "Cls_Std": cls_preds_std,
                    "Cls_True": cls_trues_mean,
                    "True_cls": trues_cls,
                    "Pred_cls": preds_cls,
                    "True_reg": trues,
                    "Pred_reg": preds,
                }
            else:
                if not os.path.exists(self.args.results_dir + '/' + self.args.experiment_name + '/'):
                    os.makedirs(os.path.dirname(self.args.results_dir + '/' + self.args.experiment_name + '/'))
                if f'({interval[0]}, {interval[1]})' not in os.listdir(
                        self.args.results_dir + '/' + self.args.experiment_name + '/'):
                    subprocess.run(
                        ["bash", "./scripts/multivariate_forecasting/Wireless/stats.sh", str(interval[0]),
                         str(interval[1])],
                        capture_output=True,
                        text=True,
                        check=True
                    )

                preds, trues = self.load_data_combined(interval)

                reg_preds_mean, reg_preds_std = self.compute_time_lagged_mean_std(preds)
                reg_trues_mean, _ = self.compute_time_lagged_mean_std(trues)

                results[f'({interval[0]}, {interval[1]})'] = {
                    "Reg_Mean": reg_preds_mean,
                    "Reg_Std": reg_preds_std,
                    "Reg_True": reg_trues_mean,
                    "True_reg": trues,
                    "Pred_reg": preds,
                }

        return results

    def build_results_dict_uniform(self):
        results = dict()
        for interval in self.train_discrete_intervals:
            if not os.path.exists(self.args.results_dir + '/' + self.args.experiment_name + '/'):
                os.makedirs(os.path.dirname(self.args.results_dir + '/' + self.args.experiment_name + '/'))
            if f'({interval[0]}, {interval[1]})' not in os.listdir(
                    self.args.results_dir + '/' + self.args.experiment_name + '/'):
                subprocess.run(
                    ["bash", "./scripts/multivariate_forecasting/Wireless/stats.sh", str(interval[0]),
                     str(interval[1])],
                    capture_output=True,
                    text=True,
                    check=True
                )

            preds, trues = self.load_data_combined(interval)

            reg_preds_mean, reg_preds_std = self.compute_time_lagged_mean_std(preds)
            reg_trues_mean, _ = self.compute_time_lagged_mean_std(trues)

            results[f'({interval[0]}, {interval[1]})'] = {
                "Reg_Mean": reg_preds_mean,
                "Reg_Std": reg_preds_std,
                "Reg_True": reg_trues_mean,
                "True_reg": trues,
                "Pred_reg": preds,
            }
        return results

    def build_results_dict_normal(self):
        interval = [0.0, 1.0]
        results = dict()
        if not os.path.exists(self.args.results_dir + '/' + self.args.experiment_name + '/'):
            os.makedirs(os.path.dirname(self.args.results_dir + '/' + self.args.experiment_name + '/'))
        if f'({interval[0]}, {interval[1]})' not in os.listdir(
                self.args.results_dir + '/' + self.args.experiment_name + '/'):
            subprocess.run(
                ["bash", "./scripts/multivariate_forecasting/Wireless/stats.sh", str(interval[0]),
                 str(interval[1])],
                capture_output=True,
                text=True,
                check=True
            )

        preds, trues = self.load_data_combined(interval)

        reg_preds_mean, reg_preds_std = self.compute_time_lagged_mean_std(preds)
        reg_trues_mean, _ = self.compute_time_lagged_mean_std(trues)

        results[f'({interval[0]}, {interval[1]})'] = {
            "Reg_Mean": reg_preds_mean,
            "Reg_Std": reg_preds_std,
            "Reg_True": reg_trues_mean,
            "True_reg": trues,
            "Pred_reg": preds,
        }
        return results

    def compute_time_lagged_mean_std(self, data):
        data = data[20:24*8, :, :]
        length = data.shape[0] + data.shape[1]
        single_curve_dict = {n: [] for n in range(length)}

        for t in range(data.shape[0]):
            row = data[t, :, self.args.feature_index]
            for offset, idx in enumerate(range(t, t + len(row))):
                single_curve_dict[idx].append(row[offset])

        mean_curve = np.array([np.mean(single_curve_dict[idx]) for idx in single_curve_dict])
        std_curve = np.array([np.std(single_curve_dict[idx]) for idx in single_curve_dict])

        return mean_curve, std_curve

    def plot_results_intervals_comp(self, results_dict, train_discrete_intervals):
        fig, ax = plt.subplots(figsize=(4, 4))

        first_model_key = next(iter(results_dict))
        x_range = len(results_dict[first_model_key]["Reg_Mean"])
        x_vals = np.arange(x_range)

        for i, (l, u) in enumerate(train_discrete_intervals):
            model_key = f"({l}, {u})"
            curve_m = results_dict[model_key]["Reg_Mean"]
            curve_std = results_dict[model_key]["Reg_Std"]
            curve_t = results_dict[model_key]["Reg_True"]

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
            if len(train_discrete_intervals) > 1:
                ax.fill_between([0, x_range], [l, l], [u, u], color=COLORS[i], alpha=0.1)

        ax.set_xlabel('Time')
        ax.set_ylabel('Forecasts')
        # ax.legend()
        # plt.xlim(min(x_vals) - 10, max(x_vals) + 10)
        # plt.ylim(-.05, 1.05)

        plt.savefig(self.save_path + 'normal_plot' + self.args.extension)

    def sliding_plot_reg(self, results_dict, train_discrete_intervals):
        sample_indices = np.arange(self.args.start_index,
                                   self.args.start_index + self.args.interval_length * 2, self.args.step_size).astype(
            int)
        feature_index = self.args.feature_index
        spacing = self.args.spacing
        fig, axs = plt.subplots(8, 1, figsize=(4, 4), sharex=True, sharey=True)

        for j, interval in enumerate(train_discrete_intervals):
            preds, trues = results_dict[f'({interval[0]}, {interval[1]})']["Pred_reg"], \
                results_dict[f'({interval[0]}, {interval[1]})']["True_reg"]

            true_color = '#1f1f1f'
            pred_color = COLORS[j]

            # ax.tick_params(axis='both', which='major')
            # ax.tick_params(axis='both', which='minor')

            for i, t in enumerate(sample_indices):
                pred_x = np.arange(t + 2 * self.args.interval_length, t + 3 * self.args.interval_length)
                y_min = interval[0]  # + spacing * i
                y_max = interval[1]  # + spacing * i

                axs[i].plot(np.arange(t, t + self.args.interval_length * 3),
                            np.array(list(trues[t, :, feature_index]) + \
                                     list(trues[t + self.args.interval_length, :, feature_index]) + \
                                     list(trues[t + 2 * self.args.interval_length, :, feature_index])),
                            # + spacing * i,
                            color=true_color, linewidth=1, label=f'True values' if j == 0 and i == 0 else "", zorder=5,
                            linestyle='--')
                axs[i].plot(np.arange(t + 2 * self.args.interval_length, t + 3 * self.args.interval_length),
                            preds[t + 2 * self.args.interval_length, :, feature_index],  # + spacing * i,
                            color=pred_color, alpha=0.9, linewidth=1,
                            label=f'Pred on ({interval[0]}, {interval[1]})' if i == 0 else "", zorder=6)
                axs[i].fill_between(pred_x, y_min, y_max, color=pred_color, alpha=0.1)
                if self.model_args.training_interval_technique == 'interval-none':
                    axs[i].plot(np.arange(t + 2 * self.args.interval_length, t + 3 * self.args.interval_length),
                                0.5 * np.ones_like(
                                    np.arange(t + 2 * self.args.interval_length, t + 3 * self.args.interval_length)),
                                # + spacing * i,
                                color='red', alpha=0.9, linewidth=1, linestyle=':', zorder=6)

            # ax.set_yticklabels(np.ones(sample_indices.size))
            # ax.set_xticklabels([])
        plt.xlabel('Time')
        # plt.tight_layout()
        fig.text(0.04, 0.5, 'Forecasts', va='center', rotation='vertical')
        # plt.legend()

        plt.savefig(self.save_path + 'sliding_plot_reg' + self.args.extension)

    def sliding_plot_cls(self):
        sample_indices = np.arange(self.args.start_index, self.args.start_index + self.args.pred_len * 2,
                                   self.args.step_size).astype(int)
        feature_index = self.args.feature_index
        spacing = self.args.spacing * 0
        fig, axs = plt.subplots(8, 1, figsize=(4, 4), sharex=True, sharey=True)

        for j, interval in enumerate(self.eval_train_discrete_intervals):
            preds, trues = self.results_dict_discrete[f'({interval[0]}, {interval[1]})']["Pred_cls"], \
                self.results_dict_discrete[f'({interval[0]}, {interval[1]})']["True_cls"]

            true_color = '#1f1f1f'
            pred_color = COLORS[j]

            # ax.tick_params(axis='both', which='major')
            # ax.tick_params(axis='both', which='minor')

            for i, t in enumerate(sample_indices):
                if self.args.true_cls_index in range(
                        len(self.eval_train_discrete_intervals)) and j == self.args.true_cls_index:
                    axs[i].plot(np.arange(t, t + self.args.interval_length * 3),
                            np.array(list(trues[t, :, feature_index]) + \
                                     list(trues[t + self.args.interval_length, :, feature_index]) + \
                                     list(trues[t + 2 * self.args.interval_length, :, feature_index])) + spacing * i,
                            color=true_color, linewidth=0.5, label=f'True values' if i == 0 else "", zorder=5,
                            linestyle='--')

                axs[i].plot(np.arange(t + 2 * self.args.interval_length, t + 3 * self.args.interval_length),
                        preds[t + 2 * self.args.interval_length, :, feature_index] + spacing * i,
                        color=pred_color, alpha=0.9, linewidth=1,
                        label=f'Pred on ({interval[0]}, {interval[1]})' if i == 0 else "", zorder=6)
            # ax.set_yticklabels([])
            # ax.set_xticklabels([])
        plt.xlabel('Time')
        # plt.tight_layout()
        fig.text(0.04, 0.5, 'Forecasts', va='center', rotation='vertical')

        plt.savefig(self.save_path + 'sliding_plot_cls' + self.args.extension)

    def plot_train_loss(self):
        title = "Training Loss"
        xlabel = "Epochs"
        ylabel = "Loss"
        epochs = np.arange(len(self.train_loss))
        plt.figure(figsize=(4, 4))
        plt.plot(epochs, self.train_loss, marker='o', linestyle='-', color='blue')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(self.save_path + 'train_loss' + self.args.extension)

    def make_bar_plots(self, flag='mae'):
        if flag == 'mae':
            idx = 0
            checkpoints_dirs = [dir for dir in os.listdir(self.args.checkpoints_dir) if 'MAE' in dir.split('_')]
        elif flag == 'mse':
            idx = 1
            checkpoints_dirs = [dir for dir in os.listdir(self.args.checkpoints_dir) if 'MSE' in dir.split('_')]

        models = ['iTransformer', 'DLinear', 'PatchTST', 'TimeMixer']
        training_methods = ['Normal', 'Uniform', 'Discrete (4)', 'Discrete (8, max)', 'Discrete (8, expectation)']

        model_colors = {
            "iTransformer": "red",
            "DLinear": "green",
            "PatchTST": "orange",
            "TimeMixer": "blue"
        }

        technique_colors = {
            "Normal": "red",
            "Uniform": "green",
            "Discrete (4)": "orange",
            "Discrete (8, max)": "blue",
            "Discrete (8, expectation)": "olive",
        }

        data = dict()
        data_df = dict()
        for model in models:
            data[model] = dict()
            data_df[model] = dict()
        for model in models:
            dir_list = [dirt for dirt in checkpoints_dirs if dirt.startswith(model)]
            for subdir in dir_list:
                if 'none' in subdir:
                    avg = 0
                    data_df[model]['Normal'] = dict()
                    for interval in self.train_discrete_intervals:
                        data_df[model]['Normal'][f'({interval[0]}, {interval[1]})'] = \
                            np.load(os.path.join(self.args.results_dir, subdir,
                                                 f'({interval[0]}, {interval[1]})' '/metrics.npy'))[idx]
                        avg += np.load(os.path.join(self.args.results_dir, subdir,
                                                    f'({interval[0]}, {interval[1]})' '/metrics.npy'))[idx]
                    avg /= self.args.nr_intervals
                    data[model]['Normal'] = avg
                    data_df[model]['Normal']['Average'] = avg
                elif 'uniform' in subdir:
                    avg = 0
                    data_df[model]['Uniform'] = dict()
                    for interval in self.train_discrete_intervals:
                        data_df[model]['Uniform'][f'({interval[0]}, {interval[1]})'] = \
                            np.load(os.path.join(self.args.results_dir, subdir,
                                                 f'({interval[0]}, {interval[1]})' '/metrics.npy'))[idx]
                        avg += np.load(os.path.join(self.args.results_dir, subdir,
                                                    f'({interval[0]}, {interval[1]})' '/metrics.npy'))[idx]
                    avg /= self.args.nr_intervals
                    data[model]['Uniform'] = avg
                    data_df[model]['Uniform']['Average'] = avg
                elif 'discrete4' in subdir:
                    avg = 0
                    data_df[model]['Discrete (4)'] = dict()
                    for interval in self.train_discrete_intervals:
                        data_df[model]['Discrete (4)'][f'({interval[0]}, {interval[1]})'] = \
                            np.load(os.path.join(self.args.results_dir, subdir,
                                                 f'({interval[0]}, {interval[1]})' '/metrics.npy'))[idx]
                        avg += np.load(os.path.join(self.args.results_dir, subdir,
                                                    f'({interval[0]}, {interval[1]})' '/metrics.npy'))[idx]
                    avg /= self.args.nr_intervals
                    data[model]['Discrete (4)'] = avg
                    data_df[model]['Discrete (4)']['Average'] = avg
                elif 'discrete8' in subdir:
                    avg = 0
                    data_df[model]['Discrete (8, max)'] = dict()
                    for interval in self.train_discrete_intervals:
                        data_df[model]['Discrete (8, max)'][f'({interval[0]}, {interval[1]})'] = \
                            np.load(os.path.join(self.args.results_dir, subdir,
                                                 f'({interval[0]}, {interval[1]})_max' '/metrics.npy'))[idx]
                        avg += np.load(os.path.join(self.args.results_dir, subdir,
                                                    f'({interval[0]}, {interval[1]})_max' '/metrics.npy'))[idx]
                    avg /= self.args.nr_intervals
                    data[model]['Discrete (8, max)'] = avg
                    data_df[model]['Discrete (8, max)']['Average'] = avg

                    avg = 0
                    data_df[model]['Discrete (8, expectation)'] = dict()
                    for interval in self.train_discrete_intervals:
                        data_df[model]['Discrete (8, expectation)'][f'({interval[0]}, {interval[1]})'] = \
                            np.load(os.path.join(self.args.results_dir, subdir,
                                                 f'({interval[0]}, {interval[1]})_expectation' '/metrics.npy'))[idx]
                        avg += np.load(os.path.join(self.args.results_dir, subdir,
                                                    f'({interval[0]}, {interval[1]})_expectation' '/metrics.npy'))[idx]
                    avg /= self.args.nr_intervals
                    data[model]['Discrete (8, expectation)'] = avg
                    data_df[model]['Discrete (8, expectation)']['Average'] = avg

        training_methods_lat = [
            r'B',
            r'U$_{0.1}$',
            r'D$_4$',
            r'D$_4^\infty$',
            r'D$_4^1$'
        ]
        for model, values in data.items():
            mae_values = [values[method] for method in training_methods]

            plt.figure(figsize=(4, 4), facecolor='white')
            ax = plt.gca()
            ax.set_facecolor('white')
            ax.spines['bottom'].set_color('black')
            ax.spines['top'].set_color('black')
            ax.spines['left'].set_color('black')
            ax.spines['right'].set_color('black')
            bars = plt.bar(training_methods_lat, mae_values, color=['blue', 'green', 'orange', 'red', 'olive'],
                           edgecolor='black',
                           linewidth=0.7, align='center')

            for bar, mae in zip(bars, mae_values):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, height, f'{mae:.4f}', ha='center', va='bottom', fontsize=8)

            plt.xlabel('Training Strategy')
            plt.ylabel('Average MAE')
            plt.ylim(0, max(mae_values) * 1.2)
            plt.grid(False)
            plt.tight_layout()
            if not os.path.exists(f'plots/average_{flag}/'):
                os.makedirs(f'plots/average_{flag}/')
            plt.savefig(f'plots/average_{flag}/{model}' + self.args.extension)

        rows = []
        for model, strategies in data_df.items():
            for training_strategy, metrics in strategies.items():
                row = {
                    "Model": model,
                    "Training Strategy": training_strategy,
                    "Average": metrics.get("Average")
                }
                for interval in self.train_discrete_intervals:
                    row[f'({interval[0]}, {interval[1]})'] = metrics.get(f'({interval[0]}, {interval[1]})')
                rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values(by=["Model", "Training Strategy"]).reset_index(drop=True)
        df.to_csv(f'plots/average_{flag}/metric_table.csv', index=False)

        rows = []
        for model, trainings in data_df.items():
            for training, metrics in trainings.items():
                intervals = [metrics[f'({interval[0]}, {interval[1]})'] for interval in self.train_discrete_intervals]
                std_val = np.std(intervals, ddof=0)
                avg_val = metrics["Average"]
                rows.append({
                    "Model": model,
                    "Training": training,
                    "Average": avg_val,
                    "Std": std_val
                })

        df = pd.DataFrame(rows)
        df["Training"] = pd.Categorical(df["Training"], categories=training_methods, ordered=True)
        df = df.sort_values(by=["Model", "Training"]).reset_index(drop=True)

        models = sorted(df["Model"].unique())
        x = np.arange(len(models))
        n_trainings = len(training_methods)
        bar_width = 0.1
        offset = (n_trainings - 1) / 2 * bar_width
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, training in enumerate(training_methods):
            subset = df[df["Training"] == training]
            avg_vals = [subset[subset["Model"] == model]["Average"].values[0] for model in models]
            std_vals = [subset[subset["Model"] == model]["Std"].values[0] for model in models]
            bar_positions = x - offset + i * bar_width
            ax.bar(bar_positions, avg_vals, width=bar_width, yerr=std_vals, capsize=5, label=training,
                   edgecolor='black',
                   linewidth=0.7, align='center', color=technique_colors[training])
        ax.set_xlabel("Model")
        ax.set_ylabel("Average MAE")
        ax.set_title("Average MAE with Standard Deviation by Model and Training Strategy")
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend(title="Training Strategy")
        plt.tight_layout()
        plt.savefig(f'plots/average_{flag}/avg_bar_plot_comp' + self.args.extension)

        x = np.arange(len(training_methods))
        bar_width = 0.2
        n_models = len(models)
        fig, ax = plt.subplots(figsize=(10, 6))
        for j, training in enumerate(training_methods):
            subset = df[df["Training"] == training].set_index("Model").reindex(models).reset_index()
            offsets = np.linspace(-bar_width * (n_models - 1) / 2, bar_width * (n_models - 1) / 2, n_models)
            for i, row in enumerate(subset.itertuples(index=False)):
                xpos = x[j] + offsets[i]
                ax.bar(xpos, row.Average, width=bar_width, yerr=row.Std,
                       color=model_colors[row.Model], edgecolor='k', capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(training_methods)
        ax.set_xlabel("Training Methodology")
        ax.set_ylabel("Average MAE")
        ax.set_title("Average MAE by Training Methodology for Each Model")
        legend_handles = [plt.Rectangle((0, 0), 1, 1, color=model_colors[model], edgecolor='k') for model in
                          models]
        ax.legend(legend_handles, models, title="Model", loc='upper right')
        plt.tight_layout()
        plt.savefig(f'plots/average_{flag}/avg_bar_plot_comp_all_in_one' + self.args.extension)

        performance = {}
        for model, trainings in data_df.items():
            values = []
            for t in training_methods:
                values.append(trainings[t]["Average"])
            performance[model] = values

        categories = training_methods
        N = len(categories)

        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], categories)
        max_val = max(max(vals) for vals in performance.values())
        plt.ylim(0, max_val * 1.1)
        radial_ticks = np.linspace(0, max_val * 1.1, 5)
        ax.set_yticks(radial_ticks)
        ax.set_yticklabels(["{:.3f}".format(tick) for tick in radial_ticks])
        for model, vals in performance.items():
            vals += vals[:1]
            ax.plot(angles, vals, linewidth=2, label=model)
            ax.fill(angles, vals, alpha=0.25)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title("Radar Plot of Average MAE Performances", size=15, y=1.1)
        plt.savefig(f'plots/average_{flag}/performance_radar_plot' + self.args.extension)


if __name__ == '__main__':
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # basic config
    parser = argparse.ArgumentParser(description="Visualization of model results")
    parser.add_argument('--experiment_name', type=str, required=True, default='Experiment',
                        help='the unique experiment name')
    parser.add_argument('--start_index', type=int, default=50, help='start index')
    parser.add_argument('--feature_index', type=int, default=0, help='feature index')
    parser.add_argument('--end_index', type=int, default=129, help='interval length index')
    parser.add_argument('--interval_length', type=int, default=24, help='interval length')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction length')
    parser.add_argument('--step_size', type=int, default=6, help='step size')
    parser.add_argument('--true_cls_index', type=int, default=0, help='true classifier index')
    parser.add_argument('--nr_intervals', type=int, default=4, help='number of intervals')
    parser.add_argument('--spacing', type=int, default=1, help='spacing between intervals')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints',
                        help='the directory to save checkpoints')
    parser.add_argument('--results_dir', type=str, default='results', help='the directory to save results')
    parser.add_argument('--plots_dir', type=str, default='plots', help='the directory to save plots')
    parser.add_argument('--extension', type=str, default='.png', help='file extension')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')

    args = parser.parse_args()
    args.use_gpu = True  # if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    viz = ModelVisualization(args=args)
    if viz.model_args.training_interval_technique == 'interval-discrete':
        # viz.plot_results_intervals_comp(viz.results_dict_discrete, viz.train_discrete_intervals)
        # viz.sliding_plot_reg(viz.results_dict_discrete, viz.train_discrete_intervals)
        # viz.sliding_plot_cls()
        # viz.plot_train_loss()
        try:
            # if viz.model_args.model == 'iTransformer':
            viz.make_bar_plots(flag='mae')
            viz.make_bar_plots(flag='mse')
        except FileNotFoundError as e:
            print('You are dumb, look what happened:', e)
    elif viz.model_args.training_interval_technique == 'interval-uniform':
        viz.plot_results_intervals_comp(viz.results_dict_uniform, viz.train_discrete_intervals)
        viz.sliding_plot_reg(viz.results_dict_uniform, viz.train_discrete_intervals)
        viz.plot_train_loss()
    else:
        viz.plot_results_intervals_comp(viz.results_dict_normal, [(0.0, 1.0)])
        viz.sliding_plot_reg(viz.results_dict_normal, [(0.0, 1.0)])
        viz.plot_train_loss()
