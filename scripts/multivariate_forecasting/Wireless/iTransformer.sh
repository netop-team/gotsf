#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate luca_timeseries

set -e
export CUDA_VISIBLE_DEVICES=1

function separate_intervals() {
  if [ $# -ne 3 ]; then
    echo "Usage: $0 delta1 delta2 nr_intervals"
    exit 1
  fi

  delta1="$1"
  delta2="$2"
  nr_intervals="$3"

  length="$(awk -v d1="$delta1" -v d2="$delta2" 'BEGIN {print d2 - d1}')"
  step="$(awk -v l="$length" -v n="$nr_intervals" 'BEGIN {print l / n}')"
  intervals=()
  for (( i=0; i<nr_intervals; i++ )); do
    start="$(awk -v d1="$delta1" -v st="$step" -v idx="$i" \
      'BEGIN {print d1 + idx * st}')"
    end="$(awk -v s="$start" -v st="$step" \
      'BEGIN {print s + st}')"
    intervals+=( "($start,$end)" )
  done

  joined="$(IFS=,; echo "${intervals[*]}")"
  echo "[$joined]"
}

delta1=-2
delta2=14
nr_intervals=5

intervals_str="$(separate_intervals $delta1 $delta2 $nr_intervals)"
intervals_no_brackets="$(echo "$intervals_str" | sed -E 's/^\[//; s/\]$//')"
parsed="$(echo "$intervals_no_brackets" | sed -E 's/\)\,\(/\)\n\(/g')"


model_name="iTransformer"
root_path="./dataset/wireless/"
data_path="fake_wireless_4_intervals.csv"
data="custom"
features="M"
seq_len=96
pred_len=24
e_layers=1
enc_in=8
dec_in=8
c_out=8
des="Exp"
d_model=32
d_ff=128
batch_size=32
itr=1

echo "Training with the normal setup on the full interval"
python -u script_train.py \
    --root_path "$root_path" \
    --data_path "$data_path" \
    --model_id "Intnone" \
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
    --batch_size "$batch_size" \
    --training_interval_technique "interval-none"

while IFS= read -r interval; do
  no_paren="$(echo "$interval" | sed -E 's/^\(//; s/\)$//; s/,/ /')"
  read -r start end <<< "$no_paren"

  echo "Training with the normal setup on interval ($start, $end)"
  python -u script_train.py \
    --root_path "$root_path" \
    --data_path "$data_path" \
    --model_id "Intspec_($start, $end)" \
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
    --batch_size "$batch_size" \
    --training_interval_technique "interval-specific" \
    --loss_delta1 "$start" \
    --loss_delta2 "$end"

done <<< "$parsed"

echo "Training with the random discrete sampling"
python -u script_train.py \
    --root_path "$root_path" \
    --data_path "$data_path" \
    --model_id "Intgen" \
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
    --batch_size "$batch_size" \
    --training_interval_technique "interval-general"

echo "Training with the random uniform sampling"
python -u script_train.py \
    --root_path "$root_path" \
    --data_path "$data_path" \
    --model_id "Intdisc" \
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
    --batch_size "$batch_size" \
    --training_interval_technique "interval-discrete"
