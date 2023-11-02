#!/bin/bash

# To get only HVs, with traffic lights evaluations
# python dqn_run.py --rv-rate 1.0 --stop-iters 1 --framework torch --num-cpu 2 --render

# To get only HVs, with no traffic lights evaluations (assuming you've changed dqn_run.py to disable traffic lights)
# python dqn_run.py --rv-rate 1.0 --stop-iters 10 --framework torch --num-cpu 2 --render

# 100% RV pen rate
python dqn_run.py --rv-rate 1.0 --stop-iters 1000 --framework torch --num-cpu 16 

# 90% RV pen rate
# python dqn_run.py --rv-rate 0.9 --stop-iters 1000 --framework torch --num-cpu 16 

# # 80% RV pen rate
# python dqn_run.py --rv-rate 0.8 --stop-iters 1000 --framework torch --num-cpu 16 

# # 70% RV pen rate
# python dqn_run.py --rv-rate 0.7 --stop-iters 1000 --framework torch --num-cpu 16

# # 60% RV pen rate
# python dqn_run.py --rv-rate 0.6 --stop-iters 1000 --framework torch --num-cpu 12 

# # 50% RV pen rate
# python dqn_run.py --rv-rate 0.5 --stop-iters 1000 --framework torch --num-cpu 2 --render 

# # 40% RV pen rate
# python dqn_run.py --rv-rate 0.4 --stop-iters 1000 --framework torch --num-cpu 16 

# # 30% RV pen rate
# python dqn_run.py --rv-rate 0.3 --stop-iters 1000 --framework torch --num-cpu 16 

# # 20% RV pen rate
# python dqn_run.py --rv-rate 0.2 --stop-iters 1000 --framework torch --num-cpu 16 

# # 10% RV pen rate
# python dqn_run.py --rv-rate 0.1 --stop-iters 1000 --framework torch --num-cpu 16 
