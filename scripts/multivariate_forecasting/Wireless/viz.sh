#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate luca_timeseries

echo "Generating plots for iTransformer"
python -u script_viz.py --experiment_name "iTransformer_none_MAE"
python -u script_viz.py --experiment_name "iTransformer_uniform_MAE" --nr_intervals 4
python -u script_viz.py --experiment_name "iTransformer_discrete4_MAE"
python -u script_viz.py --experiment_name "iTransformer_discrete8_MAE" --nr_intervals 4

echo "Generating plots for DLinear"
python -u script_viz.py --experiment_name "DLinear_none_MAE"
python -u script_viz.py --experiment_name "DLinear_uniform_MAE" --nr_intervals 4
python -u script_viz.py --experiment_name "DLinear_discrete4_MAE"
python -u script_viz.py --experiment_name "DLinear_discrete8_MAE" --nr_intervals 4

echo "Generating plots for PatchTST"
python -u script_viz.py --experiment_name "PatchTST_none_MAE"
python -u script_viz.py --experiment_name "PatchTST_uniform_MAE" --nr_intervals 4
python -u script_viz.py --experiment_name "PatchTST_discrete4_MAE"
python -u script_viz.py --experiment_name "PatchTST_discrete8_MAE" --nr_intervals 4

echo "Generating plots for TimeMixer"
python -u script_viz.py --experiment_name "TimeMixer_none_MAE"
python -u script_viz.py --experiment_name "TimeMixer_uniform_MAE" --nr_intervals 4
python -u script_viz.py --experiment_name "TimeMixer_discrete4_MAE"
python -u script_viz.py --experiment_name "TimeMixer_discrete8_MAE" --nr_intervals 4

echo "Generating plots for iTransformer"
python -u script_viz.py --experiment_name "iTransformer_none_MSE"
python -u script_viz.py --experiment_name "iTransformer_uniform_MSE" --nr_intervals 4
python -u script_viz.py --experiment_name "iTransformer_discrete4_MSE"
python -u script_viz.py --experiment_name "iTransformer_discrete8_MSE" --nr_intervals 4

echo "Generating plots for DLinear"
python -u script_viz.py --experiment_name "DLinear_none_MSE"
python -u script_viz.py --experiment_name "DLinear_uniform_MSE" --nr_intervals 4
python -u script_viz.py --experiment_name "DLinear_discrete4_MSE"
python -u script_viz.py --experiment_name "DLinear_discrete8_MSE" --nr_intervals 4

echo "Generating plots for PatchTST"
python -u script_viz.py --experiment_name "PatchTST_none_MSE"
python -u script_viz.py --experiment_name "PatchTST_uniform_MSE" --nr_intervals 4
python -u script_viz.py --experiment_name "PatchTST_discrete4_MSE"
python -u script_viz.py --experiment_name "PatchTST_discrete8_MSE" --nr_intervals 4

echo "Generating plots for TimeMixer"
python -u script_viz.py --experiment_name "TimeMixer_none_MSE"
python -u script_viz.py --experiment_name "TimeMixer_uniform_MSE" --nr_intervals 4
python -u script_viz.py --experiment_name "TimeMixer_discrete4_MSE"
python -u script_viz.py --experiment_name "TimeMixer_discrete8_MSE" --nr_intervals 4

wait