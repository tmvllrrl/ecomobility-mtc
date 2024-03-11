#!/bin/bash

# # 50% RV pen rate
python sqn_run.py --rv-rate 0.5 --stop-iters 1000 --framework torch --num-cpu 2 --render --scenario sr

# To get only HVs, with no traffic lights evaluations (assuming you've changed dqn_run.py to disable traffic lights)
# python sqn_run.py --rv-rate 0.01 --stop-iters 10 --framework torch --num-cpu 16 --render --scenario gm
