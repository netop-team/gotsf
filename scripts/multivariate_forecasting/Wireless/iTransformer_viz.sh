#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate luca_timeseries

python -u script_viz.py --experiment_name "None"
python -u script_viz.py --experiment_name "Uniform" --nr_intervals 4
python -u script_viz.py --experiment_name "Discrete4"
python -u script_viz.py --experiment_name "Discrete8" --nr_intervals 4