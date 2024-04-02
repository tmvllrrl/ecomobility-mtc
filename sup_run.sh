#!/bin/bash

# 25%, 50%, 75%, 100% RV pen rate Traffic Scenario 1
python sqn_run.py --rv-rate 0.25 --stop-iters 1000 --framework torch --num-cpu 16 --scenario one
python sqn_run.py --rv-rate 0.5 --stop-iters 1000 --framework torch --num-cpu 16 --scenario one
python sqn_run.py --rv-rate 0.75 --stop-iters 1000 --framework torch --num-cpu 16 --scenario one
python sqn_run.py --rv-rate 1.0 --stop-iters 1000 --framework torch --num-cpu 16 --scenario one

# 25%, 50%, 75%, 100% RV pen rate Traffic Scenario 2
python sqn_run.py --rv-rate 0.25 --stop-iters 1000 --framework torch --num-cpu 16 --scenario two
python sqn_run.py --rv-rate 0.5 --stop-iters 1000 --framework torch --num-cpu 16 --scenario two
python sqn_run.py --rv-rate 0.75 --stop-iters 1000 --framework torch --num-cpu 16 --scenario two
python sqn_run.py --rv-rate 1.0 --stop-iters 1000 --framework torch --num-cpu 16 --scenario two

# 25%, 50%, 75%, 100% RV pen rate Traffic Scenario 2
python sqn_run.py --rv-rate 0.25 --stop-iters 1000 --framework torch --num-cpu 16 --scenario three
python sqn_run.py --rv-rate 0.5 --stop-iters 1000 --framework torch --num-cpu 16 --scenario three
python sqn_run.py --rv-rate 0.75 --stop-iters 1000 --framework torch --num-cpu 16 --scenario three
python sqn_run.py --rv-rate 1.0 --stop-iters 1000 --framework torch --num-cpu 16 --scenario three

# 25%, 50%, 75%, 100% RV pen rate Traffic Scenario 2
python sqn_run.py --rv-rate 0.25 --stop-iters 1000 --framework torch --num-cpu 16 --scenario four
python sqn_run.py --rv-rate 0.5 --stop-iters 1000 --framework torch --num-cpu 16 --scenario four
python sqn_run.py --rv-rate 0.75 --stop-iters 1000 --framework torch --num-cpu 16 --scenario four
python sqn_run.py --rv-rate 1.0 --stop-iters 1000 --framework torch --num-cpu 16 --scenario four

# To get only HVs, with no traffic lights evaluations (assuming you've changed dqn_run.py to disable traffic lights)
# python sqn_run.py --rv-rate 0.01 --stop-iters 10 --framework torch --num-cpu 16 --render --scenario gm
