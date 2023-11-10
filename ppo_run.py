
import argparse
import os
import random

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig, PPOTorchPolicy
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
from core.custom_logger import CustomLoggerCallback


tf1, tf, tfv = try_import_tf()

parser = argparse.ArgumentParser()

parser.add_argument("--num-cpus", type=int, default=0)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=2000, help="Number of iterations to train."
)
parser.add_argument(
    "--rv-rate", type=float, default=1.0, help="RV percentage. 0.0-1.0"
)

parser.add_argument(
    "--render", action="store_true", help="Whether to render SUMO or not"
)

if __name__ == "__main__":
    args = parser.parse_args()

    ray.init(num_gpus=1, num_cpus=args.num_cpus)

    dummy_env = Env({
            "junction_list":['229','499','332','334'],
            "spawn_rl_prob":{},
            "probablity_RL":args.rv_rate,
            "cfg":'real_data/osm.sumocfg',
            "render": args.render,
            "map_xml":'real_data/CSeditClean_1.net_threelegs.xml',
            "max_episode_steps":1000,
            "conflict_mechanism":'flexible',
            "traffic_light_program":{
                "disable_state":'G',
                "disable_light_start":0
            }
        })
    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space
    dummy_env.close()
    policy = {
        "shared_policy": (
            PPOTorchPolicy,
            obs_space,
            act_space,
            None
        )}
    policy_mapping_fn = lambda agent_id, episode, worker, **kwargs: "shared_policy"
            
    config = (
        PPOConfig()
        .environment(Env, env_config={
            "junction_list":['229','499','332','334'],
            "spawn_rl_prob":{},
            "probablity_RL": args.rv_rate,
            "cfg":'real_data/osm.sumocfg',
            "render": args.render,
            "map_xml":'real_data/CSeditClean_1.net_threelegs.xml',
            # "rl_prob_range": [i*0.1 for i in range(5, 10)], # change RV penetration rate when reset
            "max_episode_steps":1000,
            "conflict_mechanism":'flexible',
            "traffic_light_program":{
                "disable_state":'G',
                "disable_light_start":0
            }
        }, 
        auto_wrap_old_gym_envs=False)
        .framework(args.framework)
        .training(
            use_critic=True,
            use_gae=True,
            kl_coeff=0.2,
            clip_param=0.3,
            lr=5e-5,
        )
        .rollouts(num_rollout_workers=args.num_cpus-1, rollout_fragment_length="auto")
        .multi_agent(policies=policy, policy_mapping_fn=policy_mapping_fn)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.ass 'ray.rllib.policy.policy_template.DQNTorchPolicy'> for PolicyID=shared_policy
        .resources(num_gpus=1, num_cpus_per_worker=1)
        .callbacks(CustomLoggerCallback)
    )

    stop = {
        "training_iteration": args.stop_iters,
    }

    results = tune.Tuner(
        "PPO", 
        param_space=config.to_dict(),
        run_config=air.RunConfig(name='PPO_RV'+str(args.rv_rate), stop=stop, verbose=3, log_to_file=True, 
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency = 5,
        )),
    ).fit()

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)
    ray.shutdown()
