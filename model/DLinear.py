import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class Model(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = self.configs.individual
        self.channels = self.configs.enc_in
        self.use_interval_classification = (self.configs.training_interval_technique == 'interval-discrete')
        self.use_deltas = (self.configs.training_interval_technique in ['interval-uniform', 'interval-discrete'])
        self.input_features = self.seq_len + 2 if self.use_deltas else self.seq_len
        self.hidden_layers_dims = [5 * self.pred_len, 3 * self.pred_len]

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            if self.use_interval_classification:
                self.MLP_Classification = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.input_features, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.input_features, self.pred_len))
                if self.use_interval_classification:
                    self.MLP_Classification.append(MLP(2 * self.seq_len + 2, self.hidden_layers_dims, self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.input_features, self.pred_len)
            self.Linear_Trend = nn.Linear(self.input_features, self.pred_len)
            if self.use_interval_classification:
                self.MLP_Classification = MLP(2 * self.seq_len + 2, self.hidden_layers_dims, self.pred_len)

            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len) * torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len) * torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x, deltas_l=None, deltas_u=None):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        class_init = torch.cat((seasonal_init, trend_init), dim=1)

        if self.use_deltas:
            if deltas_l is None and deltas_u is None:
                deltas_l = torch.tensor([self.configs.train_interval_l]).repeat(x.shape[-1])
                deltas_u = torch.tensor([self.configs.train_interval_u]).repeat(x.shape[-1])

            deltas_l = deltas_l.unsqueeze(1).unsqueeze(1)
            deltas_u = deltas_u.unsqueeze(1).unsqueeze(1)

            seasonal_init = seasonal_init.permute(0, 2, 1)
            deltas_l_seasonal = torch.tensor(deltas_l.repeat(1, seasonal_init.shape[1], 1), device=seasonal_init.device,
                                             dtype=seasonal_init.dtype)
            deltas_u_seasonal = torch.tensor(deltas_u.repeat(1, seasonal_init.shape[1], 1), device=seasonal_init.device,
                                             dtype=seasonal_init.dtype)
            seasonal_init = torch.cat((seasonal_init, deltas_l_seasonal, deltas_u_seasonal), dim=-1)

            trend_init = trend_init.permute(0, 2, 1)
            deltas_l_trend = torch.tensor(deltas_l.repeat(1, trend_init.shape[1], 1), device=trend_init.device,
                                          dtype=trend_init.dtype)
            deltas_u_trend = torch.tensor(deltas_u.repeat(1, trend_init.shape[1], 1), device=trend_init.device,
                                          dtype=trend_init.dtype)
            trend_init = torch.cat((trend_init, deltas_l_trend, deltas_u_trend), dim=-1)

            if self.use_interval_classification:
                class_init = class_init.permute(0, 2, 1)
                deltas_l_class = torch.tensor(deltas_l.repeat(1, class_init.shape[1], 1), device=class_init.device,
                                              dtype=class_init.dtype)
                deltas_u_class = torch.tensor(deltas_u.repeat(1, class_init.shape[1], 1), device=class_init.device,
                                              dtype=class_init.dtype)
                class_init = torch.cat((class_init, deltas_l_class, deltas_u_class), dim=-1)

        else:
            seasonal_init = seasonal_init.permute(0, 2, 1)
            trend_init = trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            if self.use_interval_classification:
                class_output = torch.zeros([class_init.size(0), class_init.size(1), self.pred_len],
                                       dtype=class_init.dtype).to(class_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
                if self.use_interval_classification:
                    class_output[:, i, :] = self.MLP_Classification[i](class_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
            if self.use_interval_classification:
                class_output = self.MLP_Classification(class_init)

        x = seasonal_output + trend_output
        if self.use_interval_classification:
            x, class_output = x.permute(0, 2, 1), class_output.permute(0, 2, 1)
            x = torch.cat((x, class_output), dim=1)
        else:
            x = x.permute(0, 2, 1)

        return x  # to [Batch, Output length, Channel]