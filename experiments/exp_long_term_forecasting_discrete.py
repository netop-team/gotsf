import os
import time
import pickle
import warnings
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from lion_pytorch import Lion

from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, compute_global_min_max, separate_interval
from utils.metrics import metric

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast_Discrete(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_Discrete, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.optimizer == 'lion':
            model_optim = Lion(self.model.parameters(), lr=self.args.learning_rate)
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, train_discrete_intervals):
        train_discrete_intervals = np.array(train_discrete_intervals)
        indices = np.arange(len(train_discrete_intervals))
        rand_index = np.random.choice(indices, size=self.args.batch_size)
        rand_intervals = np.array(train_discrete_intervals[rand_index])
        deltas_l = torch.tensor(rand_intervals[:, 0], device=self.device)
        deltas_u = torch.tensor(rand_intervals[:, 1], device=self.device)

        def decay_function(x, rate):
            return torch.exp(-rate * x)

        def criterion(pred, true):
            batch_size, seq_len, n_var = true.shape
            edeltas_l = deltas_l.view(batch_size, 1, 1).expand(batch_size, seq_len, n_var)
            edeltas_u = deltas_u.view(batch_size, 1, 1).expand(batch_size, seq_len, n_var)

            centre = (edeltas_u + edeltas_l) / 2
            width = (edeltas_u - edeltas_l) / 2
            signed_distance = torch.abs(true - centre) - width
            distance = torch.maximum(torch.zeros_like(true), signed_distance)

            decay_weight = decay_function(distance, self.args.decay_rate)
            if self.args.loss == 'MAE':
                loss = torch.mean(torch.abs(pred - true) * decay_weight)
            else:
                loss = torch.mean(torch.abs((pred - true) ** 2) * decay_weight)
            return loss

        return criterion, deltas_l, deltas_u

    def vali(self, vali_data, vali_loader, reg_criterion, deltas_l, deltas_u, flag):
        total_loss = []
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in (
                    tqdm(enumerate(vali_loader),
                         desc=f'Validation on {flag} with: {deltas_l[0]:.2f}, {deltas_u[0]:.2f}: ',
                         total=len(vali_loader), ncols=100)):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x, deltas_l, deltas_u)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, deltas_l, deltas_u)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, deltas_l, deltas_u)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x, deltas_l, deltas_u)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, deltas_l, deltas_u)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, deltas_l, deltas_u)

                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs[:, -(2 * self.args.pred_len):-self.args.pred_len, f_dim:]

                batch_size, seq_len, n_var = batch_y.shape
                expanded_deltas_l = deltas_l.view(batch_size, 1, 1).expand(batch_size, seq_len, n_var)
                expanded_deltas_u = deltas_u.view(batch_size, 1, 1).expand(batch_size, seq_len, n_var)
                mask = (batch_y >= expanded_deltas_l) & (batch_y <= expanded_deltas_u)
                true_eval = batch_y * mask
                pred_eval = outputs * mask
                loss = reg_criterion(pred_eval, true_eval).detach().cpu()
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        with open(os.path.join(path, 'args.pk'), 'wb') as file:
            pickle.dump(self.args, file)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        train_int_min, train_int_max = self.args.train_interval_l, self.args.train_interval_u
        train_discrete_intervals = separate_interval(
            interval=[train_int_min, train_int_max],
            nr_intervals=self.args.nr_intervals,
        )
        print(f'The model will be trained on {self.args.nr_intervals} evenly spread between {train_int_min} and {train_int_max}')
        print(f'The dataset has minimum: {compute_global_min_max([train_data])[0]} and maximum: {compute_global_min_max([train_data])[1]}')

        model_optim = self._select_optimizer()
        vali_test_criterion = nn.L1Loss()
        cls_criterion = nn.BCEWithLogitsLoss(reduction='mean')

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        train_losses = list()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in (
                    tqdm(enumerate(train_loader), desc=f'Epoch {epoch + 1}:', total=len(train_loader), ncols=100)):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                reg_criterion, deltas_l, deltas_u = self._select_criterion(train_discrete_intervals)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x, deltas_l, deltas_u)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, deltas_l, deltas_u)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, deltas_l, deltas_u)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        batch_size, seq_len, n_var = batch_y.shape
                        edeltas_l = deltas_l.view(batch_size, 1, 1).expand(batch_size, seq_len, n_var)
                        edeltas_u = deltas_u.view(batch_size, 1, 1).expand(batch_size, seq_len, n_var)
                        lower_bound, upper_bound = edeltas_l, edeltas_u
                        outputs_reg = outputs[:, -(2 * self.args.pred_len):-self.args.pred_len, f_dim:]
                        outputs_cls = outputs[:, -self.args.pred_len:, f_dim:]
                        weight = ((batch_y >= lower_bound) & (batch_y <= upper_bound)).float()
                        loss_reg = reg_criterion(outputs_reg, batch_y)
                        loss_cls = cls_criterion(outputs_cls, weight)
                        loss = loss_reg + self.args.cls_weight * loss_cls
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x, deltas_l, deltas_u)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, deltas_l, deltas_u)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, deltas_l, deltas_u)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    batch_size, seq_len, n_var = batch_y.shape
                    edeltas_l = deltas_l.view(batch_size, 1, 1).expand(batch_size, seq_len, n_var)
                    edeltas_u = deltas_u.view(batch_size, 1, 1).expand(batch_size, seq_len, n_var)
                    lower_bound, upper_bound = edeltas_l, edeltas_u
                    outputs_reg = outputs[:, -(2 * self.args.pred_len):-self.args.pred_len, f_dim:]
                    outputs_cls = outputs[:, -self.args.pred_len:, f_dim:]
                    weight = ((batch_y >= lower_bound) & (batch_y <= upper_bound)).float()
                    loss_reg = reg_criterion(outputs_reg, batch_y)
                    loss_cls = cls_criterion(outputs_cls, weight)
                    loss = loss_reg + self.args.cls_weight * loss_cls
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_losses.append(train_loss)

            vali_loss, test_loss = 0, 0
            if not self.args.enable_fast_train:
                for interval in train_discrete_intervals:
                    vali_deltas_l = torch.tensor(np.repeat(np.array([interval[0]]), len(vali_data)), device=self.device)
                    vali_deltas_u = torch.tensor(np.repeat(np.array([interval[1]]), len(vali_data)), device=self.device)
                    vali_loss += self.vali(vali_data, vali_loader, vali_test_criterion,
                                           vali_deltas_l, vali_deltas_u, flag='validation')

                    test_deltas_l = torch.tensor(np.repeat(np.array([interval[0]]), len(test_data)), device=self.device)
                    test_deltas_u = torch.tensor(np.repeat(np.array([interval[1]]), len(test_data)), device=self.device)
                    test_loss += self.vali(test_data, test_loader, vali_test_criterion,
                                           test_deltas_l, test_deltas_u, flag='test')

                vali_loss /= len(train_discrete_intervals)
                test_loss /= len(train_discrete_intervals)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        train_losses = np.array(train_losses)
        np.save(self.args.checkpoints + '/' + setting + '/' + 'train_loss.npy', train_losses)
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0, eval_on_train=False, eval_delta_l=None, eval_delta_u=None):
        test_data, test_loader = self._get_data(flag='test' if not eval_on_train else 'train-test')

        if test:
            print('Loading model...')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints + self.args.experiment_name, 'checkpoint.pth')))

        if eval_delta_l is None and eval_delta_u is None:
            eval_delta_l, eval_delta_u = compute_global_min_max([test_data])

        deltas_l = torch.tensor(np.repeat(np.array([eval_delta_l]), len(test_data)), device=self.device)
        deltas_u = torch.tensor(np.repeat(np.array([eval_delta_u]), len(test_data)), device=self.device)

        preds = list()
        trues = list()
        preds_cls = list()
        trues_cls = list()

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in (
                    tqdm(enumerate(test_loader), desc=f'Testing', total=len(test_loader), ncols=100)
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x, deltas_l, deltas_u)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, deltas_l, deltas_u)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, deltas_l, deltas_u)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x, deltas_l, deltas_u)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, deltas_l, deltas_u)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, deltas_l, deltas_u)

                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                batch_size, seq_len, n_var = batch_y.shape
                edeltas_l = deltas_l.view(batch_size, 1, 1).expand(batch_size, seq_len, n_var)
                edeltas_u = deltas_u.view(batch_size, 1, 1).expand(batch_size, seq_len, n_var)
                lower_bound, upper_bound = edeltas_l, edeltas_u
                outputs_reg = outputs[:, -(2 * self.args.pred_len):-self.args.pred_len, f_dim:]
                outputs_cls = torch.sigmoid(outputs[:, -self.args.pred_len:, f_dim:])
                weight = ((batch_y >= lower_bound) & (batch_y <= upper_bound)).float()
                preds_cls.append(outputs_cls.detach().cpu().numpy())
                trues_cls.append(weight.detach().cpu().numpy())
                outputs = outputs_reg

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        preds_cls = np.array(preds_cls)
        trues_cls = np.array(trues_cls)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        preds_cls = preds_cls.reshape(-1, preds_cls.shape[-2], preds_cls.shape[-1])
        trues_cls = trues_cls.reshape(-1, trues_cls.shape[-2], trues_cls.shape[-1])

        folder_path = self.args.results + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mask = (trues >= eval_delta_l) & (trues <= eval_delta_u)
        trues_eval = trues * mask
        preds_eval = preds * mask
        mae, mse, rmse, mape, mspe = metric(preds_eval, trues_eval)

        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'pred_cls.npy', preds_cls)
        np.save(folder_path + 'true_cls.npy', trues_cls)

        return preds, trues, preds_cls, trues_cls

