#!/bin/bash

############## Script for evaluting a trial with RV 1.0 pen rate
# python dqn_eval.py --rv-rate 1.0 --model-dir /home/michael/ray_results/DQN_RV1.0/trial/checkpoint_000310 --save-dir /home/michael/Desktop/ecomobility-mtc/eval_results --stop-timesteps 1000 

############## Script for evaluting a trial with RV 1.0 pen rate and emissions term with 0.1 coeff
# python dqn_eval.py --rv-rate 1.0 --model-dir /home/michael/ray_results/DQN_RV1.0/with_emis_term_0.1/checkpoint_000475 --save-dir /home/michael/Desktop/ecomobility-mtc/eval_results --stop-timesteps 1000 

############## Script for evaluting a trial with RV 0.9 pen rate
# python dqn_eval.py --rv-rate 0.9 --model-dir /home/michael/ray_results/DQN_RV0.9/trial/checkpoint_000270 --save-dir /home/michael/Desktop/ecomobility-mtc/eval_results --stop-timesteps 1000

############## Script for evaluting a trial with RV 0.8 pen rate
# python dqn_eval.py --rv-rate 0.8 --model-dir /home/michael/ray_results/DQN_RV0.8/trial/checkpoint_000390 --save-dir /home/michael/Desktop/ecomobility-mtc/eval_results --stop-timesteps 1000

############## Script for evaluting a trial with RV 0.7 pen rate
# python dqn_eval.py --rv-rate 0.7 --model-dir /home/michael/ray_results/DQN_RV0.7/trial/checkpoint_001000 --save-dir /home/michael/Desktop/ecomobility-mtc/eval_results --stop-timesteps 1000

############## Script for evaluting a trial with RV 0.6 pen rate
# python dqn_eval.py --rv-rate 0.6 --model-dir /home/michael/ray_results/DQN_RV0.6/trial/checkpoint_000525 --save-dir /home/michael/Desktop/ecomobility-mtc/eval_results --stop-timesteps 1000

##### ######### Script for evaluating a trial with RV 0.5 pen rate
# python dqn_eval.py --rv-rate 0.5 --model-dir /home/michael/ray_results/DQN_RV0.5/trial/checkpoint_000325 --save-dir /home/michael/Desktop/ecomobility-mtc/eval_results --stop-timesteps 1000 --render

############## Script for evaluting a trial with RV 0.4 pen rate
# python dqn_eval.py --rv-rate 0.4 --model-dir /home/michael/ray_results/DQN_RV0.4/trial/checkpoint_001000 --save-dir /home/michael/Desktop/ecomobility-mtc/eval_results --stop-timesteps 1000

############## Script for evaluting a trial with RV 0.3 pen rate
# python dqn_eval.py --rv-rate 0.3 --model-dir /home/michael/ray_results/DQN_RV0.3/trial/checkpoint_000435 --save-dir /home/michael/Desktop/ecomobility-mtc/eval_results --stop-timesteps 1000

############## Script for evaluting a trial with RV 0.2 pen rate
# python dqn_eval.py --rv-rate 0.2 --model-dir /home/michael/ray_results/DQN_RV0.2/trial/checkpoint_000575 --save-dir /home/michael/Desktop/ecomobility-mtc/eval_results --stop-timesteps 1000

############## Script for evaluting a trial with RV 0.1 pen rate
# python dqn_eval.py --rv-rate 0.1 --model-dir /home/michael/ray_results/DQN_RV0.1/trial/checkpoint_000700 --save-dir /home/michael/Desktop/ecomobility-mtc/eval_results --stop-timesteps 1000

# Evaluating SQN at goodlett midday
# python sqn_eval.py --rv-rate 0.5 --model-dir /home/michael/ray_results/DQN_RV0.5/SQN_GM/checkpoint_1000 --save-dir /home/michael/Desktop/ecomobility-mtc/eval_results --stop-timesteps 1000 --scenario gm --render

# Evaluating SQN at goodlett midday
# python sqn_eval.py --rv-rate 0.01 --model-dir /home/michael/ray_results/SQN_RV0.01gm/SQN_GM/checkpoint_1 --save-dir /home/michael/Desktop/ecomobility-mtc/eval_results --stop-timesteps 1000 --scenario gm 