from ray.rllib.algorithms.algorithm import Algorithm



from ray.rllib.algorithms.ppo import PPOConfig
import argparse
import os
import random

import ray
from ray import air, tune
from ray.rllib.algorithms.dqn import DQNConfig, DQNTorchPolicy
from env import Env
from ray.rllib.examples.models.shared_weights_model import (
    SharedWeightsModel1,
    SharedWeightsModel2,
    TF2SharedWeightsModel,
    TorchSharedWeightsModel,
)
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="DQN", help="The RLlib-registered algorithm to use."
)
parser.add_argument("--num-cpus", type=int, default=1)

parser.add_argument(
    "--stop-timesteps",
    type=int,
    default=1000,
    help="Number of timesteps to test.",
)

parser.add_argument(
    "--model-dir", type=str, required=True, help="path to the RL model for evaluation"
)
parser.add_argument(
    "--save-dir", type=str, required=True, help="folder directory for saving evaluation results"
)
parser.add_argument(
    "--rv-rate", type=float, default=1.0, help="RV percentage. 0.0-1.0"
)
parser.add_argument(
    "--explore-during-inference",
    action="store_true",
    help="Whether the trained policy should use exploration during action inference",
)

parser.add_argument(
    "--render", action="store_true", help="Whether to render SUMO or not"
)

parser.add_argument(
    "--scenario", choices=["gm", "gr", "sm", "sr"], default="gm", help="Which of the 4 traffic scenarios to train"
)

if __name__ == "__main__":
    args = parser.parse_args()

    scenario = {
        'gm': ['203789561', 'real_data/memphis/goodlett_mid/goodlett_mid.sumocfg', 'real_data/memphis/goodlett_mid/goodlett_mid.net.xml'],
        'gr': ['203789561', 'real_data/memphis/goodlett_rush/goodlett_rush.sumocfg', 'real_data/memphis/goodlett_rush/goodlett_rush.net.xml'],
        'sm': ['203926974', 'real_data/memphis/saint_mid/saint_mid.sumocfg', 'real_data/memphis/saint_mid/saint_mid.net.xml'],
        'sr': ['203926974', 'real_data/memphis/saint_rush/saint_rush.sumocfg', 'real_data/memphis/saint_rush/saint_rush.net.xml']
    }

    # ray.init(num_cpus=1)
    ray.init(local_mode= True)

    checkpoint_path = args.model_dir
    algo = Algorithm.from_checkpoint(checkpoint_path)
    
    ## TODO map xml could be parsed from sumocfg file
    env = Env({
            "junction_list":[scenario[args.scenario][0]],
            "spawn_rl_prob":{},
            "probablity_RL":args.rv_rate,
            "cfg":scenario[args.scenario][1],
            "render":args.render,
            "map_xml":scenario[args.scenario][2],
            "max_episode_steps":args.stop_timesteps,
            "conflict_mechanism":'off',
            "traffic_light_program":{
                "disable_state":'G',
                "disable_light_start":0
            }
        })
    
    episode_reward = 0
    dones = truncated = {}
    dones['__all__'] = truncated['__all__'] = False

    obs, info = env.reset()

    while not dones['__all__'] and not truncated['__all__']:
        actions = {}
        for agent_id, agent_obs in obs.items():
            actions[agent_id] = algo.compute_single_action(agent_obs, explore=args.explore_during_inference ,policy_id="shared_policy")
        obs, reward, dones, truncated, info = env.step(actions)
        for key, done in dones.items():
            if done:
                obs.pop(key)
        if dones['__all__']:
            obs, info = env.reset()
            # num_episodes += 1
    
    env.monitor.evaluate(env, save_traj=True)
    save_path = args.save_dir+'/'+str(args.rv_rate)+'log.pkl'
    env.monitor.save_to_pickle(file_name = save_path)
    algo.stop()

    ray.shutdown()