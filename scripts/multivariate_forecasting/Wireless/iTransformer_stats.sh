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

echo "Generating results for DLinear"
python -u script_stats.py --experiment_name "DLinear_none" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u"
python -u script_stats.py --experiment_name "DLinear_uniform" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u"
python -u script_stats.py --experiment_name "DLinear_discrete4" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u"
python -u script_stats.py --experiment_name "DLinear_discrete8" --nr_intervals 4 --strategy "max" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u"

echo "Generating results for iTransformer"
python -u script_stats.py --experiment_name "iTransformer_none" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u"
python -u script_stats.py --experiment_name "iTransformer_uniform" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u"
python -u script_stats.py --experiment_name "iTransformer_discrete4" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u"
python -u script_stats.py --experiment_name "iTransformer_discrete8" --nr_intervals 4 --strategy "max" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u"

echo "Generating results for iTransformer"
python -u script_stats.py --experiment_name "iTransformer_none" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u"
python -u script_stats.py --experiment_name "iTransformer_uniform" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u"
python -u script_stats.py --experiment_name "iTransformer_discrete4" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u"
python -u script_stats.py --experiment_name "iTransformer_discrete8" --nr_intervals 4 --strategy "max" --eval_interval_l "$eval_interval_l" --eval_interval_u "$eval_interval_u"

wait

