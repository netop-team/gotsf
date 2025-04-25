import pandas as pd

from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, compute_dataset_min_max, compute_global_min_max, separate_interval
# from utils.refactor_specific_utils import compute_global_min_max, separate_interval, R1Loss
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import pickle
import random
import warnings
import numpy as np
from tqdm import tqdm

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
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, loss_delta1, loss_delta2):
        return None
    #     return R1Loss(loss_weight=self.args.loss_weight,
    #                   loss_delta1=loss_delta1,
    #                   loss_delta2=loss_delta2,
    #                   push_to_boundary=self.args.push_to_boundary,
    #                   value_at_boundary=self.args.value_at_boundary)


    def sample_deltas_interval_general(self, train_int_min, train_int_max):
        dist_min_max = self.args.w_hat * (train_int_max - train_int_min)
        delta1 = (train_int_max - dist_min_max - train_int_min) * random.random() + train_int_min
        delta2 = (train_int_max - delta1 - dist_min_max) * random.random() + delta1 + dist_min_max
        if self.args.normalize_deltas:
            delta = torch.tensor([delta1 / train_int_max, delta2 / train_int_max], device=self.device)
        else:
            delta = torch.tensor([delta1, delta2], device=self.device)
        criterion = self._select_criterion(loss_delta1=delta1, loss_delta2=delta2)
        return delta, criterion

    def sample_deltas_interval_discrete(self, train_discrete_intervals, train_int_min, train_int_max):
        smp_interval = random.choice(train_discrete_intervals)
        delta1, delta2 = smp_interval[0], smp_interval[1]
        if self.args.normalize_deltas:
            delta = torch.tensor([delta1 / train_int_max, delta2 / train_int_max], device=self.device)
        else:
            delta = torch.tensor([delta1, delta2], device=self.device)
        criterion = self._select_criterion(loss_delta1=delta1, loss_delta2=delta2)
        return delta, criterion

    def sample_general_delta(self, train_int_min, train_int_max):
        dist_min_max = self.args.w_hat * (train_int_max - train_int_min)
        deltas = []
        for _ in range(self.args.batch_size):
            delta1 = (train_int_max - dist_min_max - train_int_min) * random.random() + train_int_min
            delta2 = (train_int_max - delta1 - dist_min_max) * random.random() + delta1 + dist_min_max
            deltas.append([delta1, delta2])
        deltas = np.array(deltas)
        return deltas[:, 0], deltas[:, 1]

    def vali(self, vali_data, vali_loader, criterion, delta=None):
        total_loss = []
        self.model.eval()
        delta = torch.tensor(delta, device=self.device) if delta is not None else None
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
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
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, delta)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, delta)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, delta)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, delta)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        if self.args.training_interval_technique == 'interval-specific':
            train_int_min, train_int_max = self.args.loss_delta1, self.args.loss_delta2
        else:
            train_int_min, train_int_max = compute_global_min_max([train_data])

        if self.args.training_interval_technique == 'interval-discrete':
            train_discrete_intervals = separate_interval(
                interval=[train_int_min, train_int_max],
                nr_intervals=self.args.nr_intervals
            )
            train_discrete_intervals = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
            self.train_discrete_intervals = train_discrete_intervals
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, 'args.pk'), 'wb') as file:  # Overwrites any existing file.
            pickle.dump(self.args, file)
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        vali_test_criterion = self._select_criterion(
            loss_delta1=train_int_min,
            loss_delta2=train_int_max,
        ) if self.args.training_interval_technique != 'interval-none' else nn.L1Loss()
        delta, criterion = None, vali_test_criterion

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        stats = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader),
                                                                          desc=f'Epoch {epoch + 1}:'):
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
                deltas_l = None
                deltas_u = None
                if self.args.training_interval_technique == 'interval-general':
                    delta, criterion = self.sample_deltas_interval_general(
                        train_int_min=train_int_min,
                        train_int_max=train_int_max,
                    )

                    # I = np.array(np.array(train_discrete_intervals)[
                    #                  np.random.choice(np.arange(len(train_discrete_intervals)),
                    #                                   size=self.args.batch_size)])
                    # deltas_l = I[:, 0]
                    # deltas_u = I[:, 1]
                    deltas_l, deltas_u = self.sample_general_delta(train_int_min=train_int_min,
                                                                   train_int_max=train_int_max)
                    deltas_l, deltas_u = torch.tensor(deltas_l, device=self.device), torch.tensor(deltas_u,
                                                                                                  device=self.device)



                elif self.args.training_interval_technique == 'interval-discrete':
                    delta, criterion = self.sample_deltas_interval_discrete(
                        train_discrete_intervals=train_discrete_intervals,
                        train_int_min=train_int_min,
                        train_int_max=train_int_max
                    )

                    I = np.array(np.array(train_discrete_intervals)[
                                     np.random.choice(np.arange(len(train_discrete_intervals)),
                                                      size=self.args.batch_size)])
                    deltas_l = I[:, 0]
                    deltas_u = I[:, 1]
                    deltas_l, deltas_u = torch.tensor(deltas_l, device=self.device), torch.tensor(deltas_u,
                                                                                                  device=self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, deltas_l, deltas_u)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, deltas_l, deltas_u)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, deltas_l, deltas_u)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, deltas_l, deltas_u)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    # outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    # loss = criterion(outputs, batch_y)
                    if self.args.use_interval_classification:
                        # Split [2*pred_len] into [pred_len (reg) + pred_len (cls)]
                        outputs_reg = outputs[:, -(2 * self.args.pred_len):-self.args.pred_len, f_dim:]
                        outputs_cls = outputs[:, -self.args.pred_len:, f_dim:]
                        # Classification target: 1 if batch_y in [lower_bound, upper_bound], else 0
                        batch_size, seq_len, n_var = batch_y.shape
                        edeltas_l = deltas_l.view(batch_size, 1, 1).expand(batch_size, seq_len, n_var)
                        edeltas_u = deltas_u.view(batch_size, 1, 1).expand(batch_size, seq_len, n_var)
                        weight = (((batch_y >= edeltas_l) & (batch_y <= edeltas_u))).float()
                        loss_cls = nn.BCEWithLogitsLoss(reduction='mean')(outputs_cls, weight)
                        loss = torch.mean(torch.abs(outputs_reg - batch_y) * weight) + self.args.cls_weight * loss_cls
                    else:
                        # Normal regression; just slice the final pred_len portion
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]

                        batch_size, seq_len, n_var = batch_y.shape
                        edeltas_l = deltas_l.view(batch_size, 1, 1).expand(batch_size, seq_len, n_var)
                        edeltas_u = deltas_u.view(batch_size, 1, 1).expand(batch_size, seq_len, n_var)
                        weight = (((batch_y >= edeltas_l) & (batch_y <= edeltas_u))).long()
                        loss = torch.mean(torch.abs(outputs - batch_y) * weight)


                    # loss  = criterion(outputs, batch_y)
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

            # if self.train_discrete_intervals is not None:
            #     vali_loss, test_loss = 0, 0
            #     for delta in self.train_discrete_intervals:
            #         vali_test_criterion = self._select_criterion(
            #             loss_delta1=delta[0],
            #             loss_delta2=delta[1])
            #         vali_loss += self.vali(vali_data, vali_loader, vali_test_criterion, delta)
            #         test_loss += self.vali(test_data, test_loader, vali_test_criterion, delta)
            #     vali_loss /= len(self.train_discrete_intervals)
            #     test_loss /= len(self.train_discrete_intervals)
            #
            # else:
            #     vali_loss = self.vali(vali_data, vali_loader, vali_test_criterion)
            #     test_loss = self.vali(test_data, test_loader, vali_test_criterion)
            vali_loss, test_loss = 0, 0
            stats.append([epoch, train_loss, vali_loss, test_loss])
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        pd.DataFrame(stats, columns=['Epochs', 'Tloss', 'Vloss', 'Tloss']).to_csv(path + '/' + 'stats.csv')
        return self.model

    def test(self, setting, test=0, eval_on_train=False):
        test_data, test_loader = self._get_data(flag='test' if not eval_on_train else 'train-test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        delta = None
        if (self.args.evaluate_on_interval and
                self.args.training_interval_technique in ['interval-general', 'interval-discrete']):
            if self.args.normalize_deltas:
                test_int_min, test_int_max = compute_global_min_max([test_data])
                delta = torch.tensor([self.args.loss_delta1 / test_int_max, self.args.loss_delta2 / test_int_max],
                                     device=self.device)
                deltas_l = [self.args.loss_delta1] * self.args.batch_size
                deltas_u = [self.args.loss_delta2] * self.args.batch_size

                deltas_l, deltas_u = torch.tensor(deltas_l, device=self.device), torch.tensor(deltas_u,
                                                                                              device=self.device)
            else:
                # delta = torch.tensor([self.args.loss_delta1, self.args.loss_delta2], device=self.device)
                deltas_l = [self.args.loss_delta1]
                deltas_u = [self.args.loss_delta2]

                deltas_l, deltas_u = torch.tensor(deltas_l, device=self.device), torch.tensor(deltas_u,
                                                                                              device=self.device)
        preds = []
        preds_cls = []
        trues_cls = []
        trues = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
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
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, deltas_l, deltas_u)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, deltas_l, deltas_u)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, deltas_l, deltas_u)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, deltas_l, deltas_u)

                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                # outputs = outputs[:, -self.args.pred_len:, f_dim:]


                if self.args.use_interval_classification:
                    # Split [2*pred_len] into [pred_len (reg) + pred_len (cls)]
                    outputs_reg = outputs[:, -(2 * self.args.pred_len):-self.args.pred_len, f_dim:]
                    outputs_cls = torch.sigmoid(outputs[:, -self.args.pred_len:, f_dim:]).cpu().numpy()

                    # Classification target: 1 if batch_y in [lower_bound, upper_bound], else 0
                    batch_size, seq_len, n_var = batch_y.shape
                    edeltas_l = deltas_l.view(batch_size, 1, 1).expand(batch_size, seq_len, n_var)
                    edeltas_u = deltas_u.view(batch_size, 1, 1).expand(batch_size, seq_len, n_var)
                    weight = (((batch_y >= edeltas_l) & (batch_y <= edeltas_u))).float()
                    # loss_cls = nn.BCEWithLogitsLoss(reduction='mean')(outputs_cls, weight)
                    # loss = torch.mean(torch.abs(outputs - batch_y) * weight) + self.args.cls_weight * loss_cls
                else:
                    # Normal regression; just slice the final pred_len portion
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]

                    batch_size, seq_len, n_var = batch_y.shape
                    edeltas_l = deltas_l.view(batch_size, 1, 1).expand(batch_size, seq_len, n_var)
                    edeltas_u = deltas_u.view(batch_size, 1, 1).expand(batch_size, seq_len, n_var)
                    weight = (((batch_y >= edeltas_l) & (batch_y <= edeltas_u))).long()
                    # loss = torch.mean(torch.abs(outputs - batch_y) * weight)
                outputs = outputs_reg if self.args.use_interval_classification else outputs
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                weight = weight.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                pred_cls = outputs_cls if self.args.use_interval_classification else outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                preds_cls.append(pred_cls)
                trues_cls.append(weight)

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
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if self.args.evaluate_on_interval:
            loss_delta1, loss_delta2 = self.args.loss_delta1, self.args.loss_delta2
            mask = (trues >= loss_delta1) & (trues <= loss_delta2)
            trues_eval = (trues * mask)
            preds_eval = (preds * mask)
            mae, mse, rmse, mape, mspe = metric(preds_eval, trues_eval)
        else:
            mae, mse, rmse, mape, mspe = metric(preds, trues)

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
        np.save(folder_path + 'preds_cls.npy', preds_cls)
        np.save(folder_path + 'trues_cls.npy', trues_cls)
        self.preds = preds
        self.preds_cls = preds_cls
        self.trues = trues
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
