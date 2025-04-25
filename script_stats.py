import os
import torch
import pickle
import random
import argparse
import numpy as np
from experiments.exp_long_term_forecasting_none import Exp_Long_Term_Forecast_None
from experiments.exp_long_term_forecasting_uniform import Exp_Long_Term_Forecast_Uniform
from experiments.exp_long_term_forecasting_discrete import Exp_Long_Term_Forecast_Discrete
from utils.metrics import metric
from utils.tools import get_interval_cover, separate_interval


class ModelEvaluator:
    def __init__(self, args):
        self.args = args

        if os.path.isdir(os.path.join(self.args.checkpoints_dir, self.args.experiment_name)):
            self.model_dir = os.path.join(self.args.checkpoints_dir, self.args.experiment_name)
        else:
            raise ValueError('Experiment directory does not exist')

        self.model_args = pickle.load(open(self.model_dir + '/args.pk', 'rb'))

        if self.model_args.training_interval_technique == 'interval-uniform':
            self.exp = Exp_Long_Term_Forecast_Uniform(self.model_args)
        elif self.model_args.training_interval_technique == 'interval-discrete':
            self.exp = Exp_Long_Term_Forecast_Discrete(self.model_args)
        else:
            self.exp = Exp_Long_Term_Forecast_None(self.model_args)

        train_int_min, train_int_max = self.model_args.train_interval_l, self.model_args.train_interval_u
        self.train_discrete_intervals = separate_interval(
            interval=[train_int_min, train_int_max],
            nr_intervals=self.model_args.nr_intervals,
        )

    def run_experiment_none_uniform(self):
        save_path = os.path.join(self.args.experiment_name,
                                 f'({self.args.eval_interval_l}, {self.args.eval_interval_u})')
        self.exp.test(setting=save_path, test=1, eval_on_train=self.args.eval_on_train,
                      eval_delta_l=self.args.eval_interval_l, eval_delta_u=self.args.eval_interval_u)

    def run_experiment_discrete(self):
        for interval in self.train_discrete_intervals:
            save_path = os.path.join(self.args.experiment_name, f'({interval[0]}, {interval[1]})')
            self.exp.test(setting=save_path, test=1, eval_on_train=self.args.eval_on_train,
                          eval_delta_l=interval[0], eval_delta_u=interval[1])

    def load_data_discrete(self, interval):
        results_path = os.path.join(self.args.results_dir, str(self.args.experiment_name),
                                    f'({interval[0]}, {interval[1]})')
        preds = np.load(os.path.join(results_path, 'pred.npy'))
        trues = np.load(os.path.join(results_path, 'true.npy'))
        preds_cls = np.load(os.path.join(results_path, 'pred_cls.npy'))
        trues_cls = np.load(os.path.join(results_path, 'true_cls.npy'))
        return preds, trues, preds_cls, trues_cls

    def patch_experiments_discrete(self):
        interval_cover = get_interval_cover(
            train_discrete_intervals=self.train_discrete_intervals,
            eval_delta_l=self.args.eval_interval_l,
            eval_delta_u=self.args.eval_interval_u
        )

        results = {}
        for interval in interval_cover:
            preds, trues, preds_cls, trues_cls = self.load_data_discrete(interval)

            results[f'{interval[0]},{interval[1]}'] = {
                'preds': preds,
                'trues': trues,
                'preds_cls': preds_cls,
                'trues_cls': trues_cls,
            }

        shape = trues.shape  # batch_size, sequence_length, nr_variates

        combined_preds = np.zeros_like(preds)
        combined_trues = np.zeros_like(trues)

        if self.args.strategy == 'max':
            for var_idx in range(shape[2]):
                for sample_idx in range(shape[0]):
                    for seq_idx in range(shape[1]):
                        max_cls_idx = np.argmax(
                            [results[f'{interval[0]},{interval[1]}']['preds_cls'][sample_idx, seq_idx, var_idx]
                             for interval in interval_cover])
                        selected_interval = interval_cover[max_cls_idx]
                        combined_preds[sample_idx, seq_idx, var_idx] = \
                            results[f'{selected_interval[0]},{selected_interval[1]}']['preds'][
                                sample_idx, seq_idx, var_idx]
                        combined_trues[sample_idx, seq_idx, var_idx] = \
                            results[f'{selected_interval[0]},{selected_interval[1]}']['trues'][
                                sample_idx, seq_idx, var_idx]
        elif self.args.strategy == 'expectation':
            for var_idx in range(shape[2]):
                for sample_idx in range(shape[0]):
                    for seq_idx in range(shape[1]):
                        cls_pred = np.array(
                            [results[f'{interval[0]},{interval[1]}']['preds_cls'][sample_idx, seq_idx, var_idx]
                             for interval in interval_cover]
                        )
                        reg_pred = np.array(
                            [results[f'{interval[0]},{interval[1]}']['preds'][sample_idx, seq_idx, var_idx]
                             for interval in interval_cover]
                        )
                        combined_preds[sample_idx, seq_idx, var_idx] = np.dot(cls_pred, reg_pred) / np.sum(cls_pred)

                        combined_trues[sample_idx, seq_idx, var_idx] = \
                            results[f'{interval_cover[0][0]},{interval_cover[0][1]}']['trues'][
                                sample_idx, seq_idx, var_idx]

        folder_path = self.args.results_dir + '/' + self.args.experiment_name + '/' + \
                      f'({self.args.eval_interval_l}, {self.args.eval_interval_u})/'
        folder_path_additional = self.args.results_dir + '/' + self.args.experiment_name + '/' + \
                      f'({self.args.eval_interval_l}, {self.args.eval_interval_u})_{self.args.strategy}/'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if not os.path.exists(folder_path_additional):
            os.makedirs(folder_path_additional)

        np.save(folder_path + 'pred.npy', combined_preds)
        np.save(folder_path + 'true.npy', combined_trues)
        np.save(folder_path_additional + 'pred.npy', combined_preds)
        np.save(folder_path_additional + 'true.npy', combined_trues)

        mask = (combined_trues >= self.args.eval_interval_l) & (combined_trues <= self.args.eval_interval_u)
        trues_eval = combined_trues * mask
        preds_eval = combined_preds * mask
        mae, mse, rmse, mape, mspe = metric(preds_eval, trues_eval)
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path_additional + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        print('Combined predictions - mse:{}, mae:{}'.format(mse, mae))

        return


if __name__ == '__main__':
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='iTransformer')

    # basic config
    parser.add_argument('--experiment_name', type=str, required=True, default='Experiment',
                        help='the unique experiment name')
    parser.add_argument('--eval_interval_l', type=float, default=0)
    parser.add_argument('--eval_interval_u', type=float, default=0)
    parser.add_argument('--eval_on_train', type=bool, default=False, help="evaluate on the train set")
    parser.add_argument('--strategy', type=str, default='max', choices=['max', 'expectation'],
                        help='the strategy used for training')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='the directory to save checkpoints')
    parser.add_argument('--results_dir', type=str, default='results', help='the directory to save results')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')

    args = parser.parse_args()
    args.use_gpu = True  # if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    model_evaluator = ModelEvaluator(args)

    if model_evaluator.model_args.training_interval_technique == 'interval-discrete':
        path = args.results_dir + '/' + args.experiment_name + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        interval_directories = os.listdir(path)
        if len(interval_directories) == 0:
            model_evaluator.run_experiment_discrete()
        if args.eval_interval_l != args.eval_interval_u:
            if [args.eval_interval_l, args.eval_interval_u] not in model_evaluator.train_discrete_intervals:
                model_evaluator.patch_experiments_discrete()
    else:
        if args.eval_interval_l != args.eval_interval_u:
            model_evaluator.run_experiment_none_uniform()
