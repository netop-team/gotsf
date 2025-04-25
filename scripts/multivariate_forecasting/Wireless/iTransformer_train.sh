#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate luca_timeseries

set -e
export CUDA_VISIBLE_DEVICES=1

model_name="iTransformer"
root_path="./dataset/wireless/"
data_path_noise="toy_example_random_noise_0.05_N_1_T_3072.csv"
data_path_noiseless="toy_example_random_noise_0.0_N_1_T_3072.csv"
data="custom"
features="M"
seq_len=48
pred_len=24
e_layers=2
enc_in=1
dec_in=1
c_out=1
des="Exp"
d_model=128
d_ff=128
n_heads=4
batch_size=32
itr=1

python -u script_train.py \
    --root_path "$root_path" \
    --data_path "$data_path_noise" \
    --experiment_name "None" \
    --model_id "None" \
    --model "$model_name" \
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
    --learning_rate 0.001 \
    --training_interval_technique "interval-none" \
    --dropout 0.1 \
    --train_epochs 30

python -u script_train.py \
    --root_path "$root_path" \
    --data_path "$data_path_noise" \
    --model_id "Uniform" \
    --experiment_name "Uniform" \
    --model "$model_name" \
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
    --learning_rate 0.001 \
    --training_interval_technique "interval-uniform" \
    --dropout 0.1 \
    --w_hat 0.1 \
    --decay_rate 1000000000000 \
    --train_epochs 50 \
    --train_interval_l 0 \
    --train_interval_u 1

python -u script_train.py \
    --root_path "$root_path" \
    --data_path "$data_path_noise" \
    --experiment_name "Discrete4" \
    --model_id "Discrete" \
    --model "$model_name" \
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
    --learning_rate 0.001 \
    --training_interval_technique "interval-discrete" \
    --dropout 0.1 \
    --decay_rate 37 \
    --train_epochs 60 \
    --nr_intervals 4 \
    --train_interval_l 0 \
    --train_interval_u 1

python -u script_train.py \
    --root_path "$root_path" \
    --data_path "$data_path_noise" \
    --experiment_name "Discrete8" \
    --model_id "Discrete" \
    --model "$model_name" \
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
    --learning_rate 0.001 \
    --training_interval_technique "interval-discrete" \
    --dropout 0.1 \
    --decay_rate 37 \
    --train_epochs 60 \
    --nr_intervals 8 \
    --train_interval_l 0 \
    --train_interval_u 1

wait