### Run Configuration for `run.py` (under construction...)

This document outlines the configuration for running the `run.py` script.

**Base Configuration**

```
python3 run.py  --is_training 1 --root_path ./dataset/wireless/ --data_path wireless.csv --model_id wireless_96_24_hase --model iTransformer --data custom --features M --seq_len 96 --pred_len 24 --e_layers 1 --enc_in 8 --dec_in 8 --c_out 8 --des Exp --d_model 32 --d_ff 128 --itr 1 --batch_size 32
```

**Training Modes**

* **Normal Training:** Set `--use_deltas` to 0 (default).
* **Fixed Interval Training:** Set `--use_deltas` to 1 and define the intervals using hyperparameters.
* **Dynamic Interval Training (Uniform Distribution):** Set `--use_deltas` to 2.
* **Dynamic Interval Training (Fixed Interval):** Set `--use_deltas` to 3.

**Evaluation Options**

* `--loss_delta1, --loss_delta2`: Define the start and end points of the evaluation interval.
* `--evaluate_on_interval`: Set to `True` to evaluate the test set on the specified interval.
* `--loss_weight_less_importance`: Weight for the loss outside the evaluation interval (less important).
* `--w_hat`: Minimum distance of the interval for dynamic training with uniform distribution (use_deltas=2).
* `--interval_length`: The number of intervals you want to have in the interval list for random sampling (use_deltas=3)

**File Paths**

* Checkpoints: Saved to the `/checkpoints` directory.
* Results: Predictions, true values, and metrics are saved to the `/results` directory.

**Additional Notes**

* The script utilizes the `iTransformer` model with custom data (`--data custom`).
* Features are specified by `--features M`.
* Sequence length (`--seq_len`) is set to 96 and prediction length (`--pred_len`) is set to 24.

========

* 1 script to execute baseline model with uniform importance over target values
* 5 scripts to execute interval-specific models. [min, min + (max-min)/ 5] is one
  model, [min+ (max-min)/ 5, min + 2 *(max-min)/ 5] is another model
* 1 script to execute randomly sampled intervals model. w.p. 1/6 it will select one of the intervals   [min, min + (max-min)/ 5],.... etc.
* 1 script to execute randomly sample interval model u.a.r.
* Total: 8 script. Put in same file (ending with .sh). one script after another running sequentially. 
