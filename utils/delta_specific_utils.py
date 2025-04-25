import numpy as np
import torch


def R1Loss(loss_weight, loss_delta1, loss_delta2):
    if loss_delta1 > loss_delta2:
        loss_delta1, loss_delta2 = loss_delta2, loss_delta1

    def criterion(pred, true):
        weight = 1 - (~((true >= loss_delta1) & (true <= loss_delta2))).long() * (1 - loss_weight)
        loss = torch.mean(torch.abs(true - pred) * weight)
        return loss

    return criterion

def compute_dataset_min_max(dataset):
    ds_min, ds_max = float('inf'), float('-inf')
    for i in range(len(dataset)):
        x_np, y_np = dataset[i][0], dataset[i][1]
        combined = np.concatenate([x_np.flatten(), y_np.flatten()])
        ds_min = min(ds_min, np.min(combined))
        ds_max = max(ds_max, np.max(combined))
    return ds_min, ds_max


def compute_global_min_max(datasets):
    global_min, global_max = float('inf'), float('-inf')
    for ds in datasets:
        ds_min, ds_max = compute_dataset_min_max(ds)
        global_min = min(global_min, ds_min)
        global_max = max(global_max, ds_max)
    return float(global_min), float(global_max)


def separate_interval(interval, nr_intervals):
        attention_intervals = list()

        codomain_length = interval[1] - interval[0]
        eps = codomain_length / nr_intervals
        delta1, delta2 = interval[0], interval[0] + eps

        attention_intervals.append((interval[0], interval[1]))
        for i in range(nr_intervals):
            attention_intervals.append((delta1, delta2))
            delta1, delta2 = delta1 + eps, delta2 + eps

        return attention_intervals
