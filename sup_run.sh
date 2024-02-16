#!/bin/bash

# # 50% RV pen rate
python sqn_run.py --rv-rate 0.5 --stop-iters 1000 --framework torch --num-cpu 16 --render --scenario gm
