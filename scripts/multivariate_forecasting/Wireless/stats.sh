#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate luca_timeseries

if [ "$#" -ne 2 ]; then
  eval_interval_l=0
  eval_interval_u=0
else
  eval_interval_l="$1"
  eval_interval_u="$2"
fi

echo "Generating results for iTransformer"

echo "Generating results for iTransformer-none"
python -u script_stats.py --experiment_name "iTransformer_none_MSE" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

echo "Generating results for iTransformer-uniform"
python -u script_stats.py --experiment_name "iTransformer_uniform_MSE" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

echo "Generating results for iTransformer-discrete4"
python -u script_stats.py --experiment_name "iTransformer_discrete4_MSE" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

echo "Generating results for iTransformer-discrete8"
python -u script_stats.py --experiment_name "iTransformer_discrete8_MSE" --strategy "max" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

echo "Generating results for DLinear"

echo "Generating results for DLinear-none"
python -u script_stats.py --experiment_name "DLinear_none_MSE" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

echo "Generating results for DLinear-uniform"
python -u script_stats.py --experiment_name "DLinear_uniform_MSE" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

echo "Generating results for DLinear-discrete4"
python -u script_stats.py --experiment_name "DLinear_discrete4_MSE" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

echo "Generating results for DLinear-discrete8"
python -u script_stats.py --experiment_name "DLinear_discrete8_MSE" --strategy "max" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

echo "Generating results for PatchTST"

echo "Generating results for PatchTST-none"
python -u script_stats.py --experiment_name "PatchTST_none_MSE" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

echo "Generating results for PatchTST-uniform"
python -u script_stats.py --experiment_name "PatchTST_uniform_MSE" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

echo "Generating results for PatchTS-discrete4"
python -u script_stats.py --experiment_name "PatchTST_discrete4_MSE" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

echo "Generating results for PatchTST-discrete8"
python -u script_stats.py --experiment_name "PatchTST_discrete8_MSE" --strategy "max" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

echo "Generating results for TimeMixer"

echo "Generating results for TimeMixer-none"
python -u script_stats.py --experiment_name "TimeMixer_none_MSE" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

echo "Generating results for TimeMixer-uniform"
python -u script_stats.py --experiment_name "TimeMixer_uniform_MSE" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

echo "Generating results for TimeMixer-discrete4"
python -u script_stats.py --experiment_name "TimeMixer_discrete4_MSE" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

echo "Generating results for TimeMixer-discrete8"
python -u script_stats.py --experiment_name "TimeMixer_discrete8_MSE" --strategy "max" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

wait

echo "Generating results for iTransformer"

echo "Generating results for iTransformer-none"
python -u script_stats.py --experiment_name "iTransformer_none_MAE" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

echo "Generating results for iTransformer-uniform"
python -u script_stats.py --experiment_name "iTransformer_uniform_MAE" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

echo "Generating results for iTransformer-discrete4"
python -u script_stats.py --experiment_name "iTransformer_discrete4_MAE" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

echo "Generating results for iTransformer-discrete8"
python -u script_stats.py --experiment_name "iTransformer_discrete8_MAE" --strategy "max" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

echo "Generating results for DLinear"

echo "Generating results for DLinear-none"
python -u script_stats.py --experiment_name "DLinear_none_MAE" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

echo "Generating results for DLinear-uniform"
python -u script_stats.py --experiment_name "DLinear_uniform_MAE" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

echo "Generating results for DLinear-discrete4"
python -u script_stats.py --experiment_name "DLinear_discrete4_MAE" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

echo "Generating results for DLinear-discrete8"
python -u script_stats.py --experiment_name "DLinear_discrete8_MAE" --strategy "max" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

echo "Generating results for PatchTST"

echo "Generating results for PatchTST-none"
python -u script_stats.py --experiment_name "PatchTST_none_MAE" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

echo "Generating results for PatchTST-uniform"
python -u script_stats.py --experiment_name "PatchTST_uniform_MAE" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

echo "Generating results for PatchTS-discrete4"
python -u script_stats.py --experiment_name "PatchTST_discrete4_MAE" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

echo "Generating results for PatchTST-discrete8"
python -u script_stats.py --experiment_name "PatchTST_discrete8_MAE" --strategy "max" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

echo "Generating results for TimeMixer"

echo "Generating results for TimeMixer-none"
python -u script_stats.py --experiment_name "TimeMixer_none_MAE" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

echo "Generating results for TimeMixer-uniform"
python -u script_stats.py --experiment_name "TimeMixer_uniform_MAE" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

echo "Generating results for TimeMixer-discrete4"
python -u script_stats.py --experiment_name "TimeMixer_discrete4_MAE" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

echo "Generating results for TimeMixer-discrete8"
python -u script_stats.py --experiment_name "TimeMixer_discrete8_MAE" --strategy "max" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

wait

python -u script_stats.py --experiment_name "iTransformer_discrete8_MSE" --strategy "expectation" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &
python -u script_stats.py --experiment_name "DLinear_discrete8_MSE" --strategy "expectation" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &
python -u script_stats.py --experiment_name "PatchTST_discrete8_MSE" --strategy "expectation" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &
python -u script_stats.py --experiment_name "TimeMixer_discrete8_MSE" --strategy "expectation" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

python -u script_stats.py --experiment_name "iTransformer_discrete8_MAE" --strategy "expectation" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &
python -u script_stats.py --experiment_name "DLinear_discrete8_MAE" --strategy "expectation" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &
python -u script_stats.py --experiment_name "PatchTST_discrete8_MAE" --strategy "expectation" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &
python -u script_stats.py --experiment_name "TimeMixer_discrete8_MAE" --strategy "expectation" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u" &

wait