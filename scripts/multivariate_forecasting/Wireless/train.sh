#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate luca_timeseries

set -e
export CUDA_VISIBLE_DEVICES=1

root_path="./dataset/wireless/"
data_path_noise="wireless.csv"
data="custom"
features="M"
seq_len=96
pred_len=24
e_layers=2
enc_in=100
dec_in=100
c_out=100
des="Exp"
d_model=128
d_ff=128
n_heads=4
batch_size=32
itr=1
loss="$1"

echo "Starting training for iTransformer"
python -u script_train.py \
    --root_path "$root_path" \
    --data_path "$data_path_noise" \
    --experiment_name "iTransformer_none_${loss}" \
    --model_id "None" \
    --model "iTransformer" \
    --data "$data" \
    --features "$features" \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --e_layers "$e_layers" \
    --enc_in "$enc_in" \
    --dec_in "$dec_in" \
    --c_out "$c_out" \
    --des "$des" \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --itr "$itr" \
    --n_heads "$n_heads" \
    --batch_size "$batch_size" \
    --lradj type3 \
    --learning_rate 0.0001 \
    --training_interval_technique "interval-none" \
    --dropout 0.1 \
    --train_epochs 60 \
    --loss "${loss}" &

python -u script_train.py \
    --root_path "$root_path" \
    --data_path "$data_path_noise" \
    --experiment_name "iTransformer_uniform_${loss}" \
    --model_id "Uniform" \
    --model "iTransformer" \
    --data "$data" \
    --features "$features" \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --e_layers "$e_layers" \
    --enc_in "$enc_in" \
    --dec_in "$dec_in" \
    --c_out "$c_out" \
    --des "$des" \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --itr "$itr" \
    --n_heads "$n_heads" \
    --batch_size "$batch_size" \
    --lradj type3 \
    --learning_rate 0.0001 \
    --training_interval_technique "interval-uniform" \
    --dropout 0.1 \
    --w_hat 0.1 \
    --train_epochs 60 \
    --train_interval_l 0 \
    --train_interval_u 8 \
    --loss "${loss}" &

python -u script_train.py \
    --root_path "$root_path" \
    --data_path "$data_path_noise" \
    --experiment_name "iTransformer_discrete4_${loss}" \
    --model_id "Discrete" \
    --model "iTransformer" \
    --data "$data" \
    --features "$features" \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --e_layers "$e_layers" \
    --enc_in "$enc_in" \
    --dec_in "$dec_in" \
    --c_out "$c_out" \
    --des "$des" \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --itr "$itr" \
    --n_heads "$n_heads" \
    --batch_size "$batch_size" \
    --lradj type3 \
    --learning_rate 0.0001 \
    --training_interval_technique "interval-discrete" \
    --dropout 0.1 \
    --decay_rate 37 \
    --train_epochs 60 \
    --nr_intervals 8 \
    --train_interval_l 0 \
    --train_interval_u 8 \
    --loss "${loss}" &

python -u script_train.py \
    --root_path "$root_path" \
    --data_path "$data_path_noise" \
    --experiment_name "iTransformer_discrete8_${loss}" \
    --model_id "Discrete" \
    --model "iTransformer" \
    --data "$data" \
    --features "$features" \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --e_layers "$e_layers" \
    --enc_in "$enc_in" \
    --dec_in "$dec_in" \
    --c_out "$c_out" \
    --des "$des" \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --itr "$itr" \
    --n_heads "$n_heads" \
    --batch_size "$batch_size" \
    --lradj type3 \
    --learning_rate 0.0001 \
    --training_interval_technique "interval-discrete" \
    --dropout 0.1 \
    --decay_rate 37 \
    --train_epochs 60 \
    --nr_intervals 16 \
    --train_interval_l 0 \
    --train_interval_u 8 \
    --loss "${loss}" &

echo "Starting training for DLinear"
python -u script_train.py \
    --root_path "$root_path" \
    --data_path "$data_path_noise" \
    --experiment_name "DLinear_none_${loss}" \
    --model_id "None" \
    --model "DLinear" \
    --data "$data" \
    --features "$features" \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --e_layers "$e_layers" \
    --enc_in "$enc_in" \
    --dec_in "$dec_in" \
    --c_out "$c_out" \
    --des "$des" \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --itr "$itr" \
    --n_heads "$n_heads" \
    --batch_size "$batch_size" \
    --lradj type3 \
    --learning_rate 0.0001 \
    --training_interval_technique "interval-none" \
    --dropout 0.1 \
    --train_epochs 60 \
    --loss "${loss}" &

python -u script_train.py \
    --root_path "$root_path" \
    --data_path "$data_path_noise" \
    --experiment_name "DLinear_uniform_${loss}" \
    --model_id "Uniform" \
    --model "DLinear" \
    --data "$data" \
    --features "$features" \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --e_layers "$e_layers" \
    --enc_in "$enc_in" \
    --dec_in "$dec_in" \
    --c_out "$c_out" \
    --des "$des" \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --itr "$itr" \
    --n_heads "$n_heads" \
    --batch_size "$batch_size" \
    --lradj type3 \
    --learning_rate 0.0001 \
    --training_interval_technique "interval-uniform" \
    --dropout 0.1 \
    --w_hat 0.1 \
    --train_epochs 60 \
    --train_interval_l 0 \
    --train_interval_u 8 \
    --loss "${loss}" &

python -u script_train.py \
    --root_path "$root_path" \
    --data_path "$data_path_noise" \
    --experiment_name "DLinear_discrete4_${loss}" \
    --model_id "Discrete" \
    --model "DLinear" \
    --data "$data" \
    --features "$features" \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --e_layers "$e_layers" \
    --enc_in "$enc_in" \
    --dec_in "$dec_in" \
    --c_out "$c_out" \
    --des "$des" \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --itr "$itr" \
    --n_heads "$n_heads" \
    --batch_size "$batch_size" \
    --lradj type3 \
    --learning_rate 0.0001 \
    --training_interval_technique "interval-discrete" \
    --dropout 0.1 \
    --decay_rate 37 \
    --train_epochs 60 \
    --nr_intervals 8 \
    --train_interval_l 0 \
    --train_interval_u 8 \
    --loss "${loss}" &

python -u script_train.py \
    --root_path "$root_path" \
    --data_path "$data_path_noise" \
    --experiment_name "DLinear_discrete8_${loss}" \
    --model_id "Discrete" \
    --model "DLinear" \
    --data "$data" \
    --features "$features" \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --e_layers "$e_layers" \
    --enc_in "$enc_in" \
    --dec_in "$dec_in" \
    --c_out "$c_out" \
    --des "$des" \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --itr "$itr" \
    --n_heads "$n_heads" \
    --batch_size "$batch_size" \
    --lradj type3 \
    --learning_rate 0.0001 \
    --training_interval_technique "interval-discrete" \
    --dropout 0.1 \
    --decay_rate 37 \
    --train_epochs 60 \
    --nr_intervals 16 \
    --train_interval_l 0 \
    --train_interval_u 8 \
    --loss "${loss}" &

echo "Starting training for PatchTST"
python -u script_train.py \
    --root_path "$root_path" \
    --data_path "$data_path_noise" \
    --experiment_name "PatchTST_none_${loss}" \
    --model_id "None" \
    --model "PatchTST" \
    --data "$data" \
    --features "$features" \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --e_layers "$e_layers" \
    --enc_in "$enc_in" \
    --dec_in "$dec_in" \
    --c_out "$c_out" \
    --des "$des" \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --itr "$itr" \
    --n_heads "$n_heads" \
    --batch_size "$batch_size" \
    --lradj type3 \
    --learning_rate 0.0001 \
    --training_interval_technique "interval-none" \
    --dropout 0.1 \
    --train_epochs 60 \
    --loss "${loss}" &

python -u script_train.py \
    --root_path "$root_path" \
    --data_path "$data_path_noise" \
    --experiment_name "PatchTST_uniform_${loss}" \
    --model_id "Uniform" \
    --model "PatchTST" \
    --data "$data" \
    --features "$features" \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --e_layers "$e_layers" \
    --enc_in "$enc_in" \
    --dec_in "$dec_in" \
    --c_out "$c_out" \
    --des "$des" \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --itr "$itr" \
    --n_heads "$n_heads" \
    --batch_size "$batch_size" \
    --lradj type3 \
    --learning_rate 0.0001 \
    --training_interval_technique "interval-uniform" \
    --dropout 0.1 \
    --w_hat 0.1 \
    --train_epochs 60 \
    --train_interval_l 0 \
    --train_interval_u 8 \
    --loss "${loss}" &

python -u script_train.py \
    --root_path "$root_path" \
    --data_path "$data_path_noise" \
    --experiment_name "PatchTST_discrete4_${loss}" \
    --model_id "Discrete" \
    --model "PatchTST" \
    --data "$data" \
    --features "$features" \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --e_layers "$e_layers" \
    --enc_in "$enc_in" \
    --dec_in "$dec_in" \
    --c_out "$c_out" \
    --des "$des" \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --itr "$itr" \
    --n_heads "$n_heads" \
    --batch_size "$batch_size" \
    --lradj type3 \
    --learning_rate 0.0001 \
    --training_interval_technique "interval-discrete" \
    --dropout 0.1 \
    --decay_rate 37 \
    --train_epochs 60 \
    --nr_intervals 8 \
    --train_interval_l 0 \
    --train_interval_u 8 \
    --loss "${loss}" &

python -u script_train.py \
    --root_path "$root_path" \
    --data_path "$data_path_noise" \
    --experiment_name "PatchTST_discrete8_${loss}" \
    --model_id "Discrete" \
    --model "PatchTST" \
    --data "$data" \
    --features "$features" \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --e_layers "$e_layers" \
    --enc_in "$enc_in" \
    --dec_in "$dec_in" \
    --c_out "$c_out" \
    --des "$des" \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --itr "$itr" \
    --n_heads "$n_heads" \
    --batch_size "$batch_size" \
    --lradj type3 \
    --learning_rate 0.0001 \
    --training_interval_technique "interval-discrete" \
    --dropout 0.1 \
    --decay_rate 37 \
    --train_epochs 60 \
    --nr_intervals 16 \
    --train_interval_l 0 \
    --train_interval_u 8 \
    --loss "${loss}" &

echo "Starting training for TimeMixer"
python -u script_train.py \
    --root_path "$root_path" \
    --data_path "$data_path_noise" \
    --experiment_name "TimeMixer_none_${loss}" \
    --model_id "None" \
    --model "TimeMixer" \
    --data "$data" \
    --features "$features" \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --e_layers "$e_layers" \
    --enc_in "$enc_in" \
    --dec_in "$dec_in" \
    --c_out "$c_out" \
    --des "$des" \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --itr "$itr" \
    --n_heads "$n_heads" \
    --batch_size "$batch_size" \
    --lradj type3 \
    --learning_rate 0.0001 \
    --training_interval_technique "interval-none" \
    --dropout 0.1 \
    --train_epochs 60 \
    --loss "${loss}" &

python -u script_train.py \
    --root_path "$root_path" \
    --data_path "$data_path_noise" \
    --experiment_name "TimeMixer_uniform_${loss}" \
    --model_id "Uniform" \
    --model "TimeMixer" \
    --data "$data" \
    --features "$features" \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --e_layers "$e_layers" \
    --enc_in "$enc_in" \
    --dec_in "$dec_in" \
    --c_out "$c_out" \
    --des "$des" \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --itr "$itr" \
    --n_heads "$n_heads" \
    --batch_size "$batch_size" \
    --lradj type3 \
    --learning_rate 0.0001 \
    --training_interval_technique "interval-uniform" \
    --dropout 0.1 \
    --w_hat 0.1 \
    --train_epochs 60 \
    --train_interval_l 0 \
    --train_interval_u 8 \
    --loss "${loss}" &

python -u script_train.py \
    --root_path "$root_path" \
    --data_path "$data_path_noise" \
    --experiment_name "TimeMixer_discrete4_${loss}" \
    --model_id "Discrete" \
    --model "TimeMixer" \
    --data "$data" \
    --features "$features" \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --e_layers "$e_layers" \
    --enc_in "$enc_in" \
    --dec_in "$dec_in" \
    --c_out "$c_out" \
    --des "$des" \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --itr "$itr" \
    --n_heads "$n_heads" \
    --batch_size "$batch_size" \
    --lradj type3 \
    --learning_rate 0.0001 \
    --training_interval_technique "interval-discrete" \
    --dropout 0.1 \
    --decay_rate 37 \
    --train_epochs 60 \
    --nr_intervals 8 \
    --train_interval_l 0 \
    --train_interval_u 8 \
    --loss "${loss}" &

python -u script_train.py \
    --root_path "$root_path" \
    --data_path "$data_path_noise" \
    --experiment_name "TimeMixer_discrete8_${loss}" \
    --model_id "Discrete" \
    --model "TimeMixer" \
    --data "$data" \
    --features "$features" \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --e_layers "$e_layers" \
    --enc_in "$enc_in" \
    --dec_in "$dec_in" \
    --c_out "$c_out" \
    --des "$des" \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --itr "$itr" \
    --n_heads "$n_heads" \
    --batch_size "$batch_size" \
    --lradj type3 \
    --learning_rate 0.0001 \
    --training_interval_technique "interval-discrete" \
    --dropout 0.1 \
    --decay_rate 37 \
    --train_epochs 60 \
    --nr_intervals 16 \
    --train_interval_l 0 \
    --train_interval_u 8 \
    --loss "${loss}" &

wait