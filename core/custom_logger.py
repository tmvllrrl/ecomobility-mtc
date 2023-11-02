"""Example of using RLlib's debug callbacks.

Here we use callbacks to track the average CartPole pole angle magnitude as a
custom metric.
"""

from typing import Dict
import argparse
import numpy as np

import ray
from ray import tune
from ray import train
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.algorithms.callbacks import DefaultCallbacks


class CustomLoggerCallback(DefaultCallbacks):
    def on_episode_start(
            self,
            *,
            worker,
            base_env,
            policies,
            episode,
            env_index = None,
            **kwargs,
        ):
        episode.user_data["conflict_rate"] = []
        episode.user_data["avg_wait"] = []
        episode.user_data["avg_fuel"] = []
        episode.user_data["avg_co2_emissions"] = []
        episode.user_data["avg_co_emissions"] = []
        episode.user_data["avg_hc_emissions"] = []
        episode.user_data["avg_nox_emissions"] = []

    def on_episode_step(
            self,
            *,
            worker,
            base_env,
            policies = None,
            episode,
            env_index= None,
            **kwargs,
        ):
        conflict_rate = worker.env.monitor.conflict_rate[-1]
        episode.user_data["conflict_rate"].extend([conflict_rate])
        total_wait = 0
        for id in worker.env.previous_global_waiting.keys():
            total_wait += worker.env.previous_global_waiting[id]['sum']
        episode.user_data["avg_wait"].extend([total_wait])

        avg_fuel = worker.env.monitor.overall_fuel_record[-1]
        episode.user_data["avg_fuel"].extend([avg_fuel])

        avg_co2_emissions = worker.env.monitor.overall_co2_record[-1]
        episode.user_data["avg_co2_emissions"].extend([avg_co2_emissions])

        avg_co_emissions = worker.env.monitor.overall_co_record[-1]
        episode.user_data["avg_co_emissions"].extend([avg_co_emissions])

        avg_hc_emissions = worker.env.monitor.overall_hc_record[-1]
        episode.user_data["avg_hc_emissions"].extend([avg_hc_emissions])

        avg_nox_emissions = worker.env.monitor.overall_nox_record[-1]
        episode.user_data["avg_nox_emissions"].extend([avg_nox_emissions])

    def on_episode_end(
        self,
        *,
        worker,
        base_env,
        policies,
        episode,
        env_index = None,
        **kwargs,
    ):
        episode.custom_metrics["conflict_rate"] = np.mean(episode.user_data["conflict_rate"])
        episode.custom_metrics["avg_wait"] = np.mean(episode.user_data["avg_wait"])
        episode.custom_metrics["avg_fuel"] = np.mean(episode.user_data["avg_fuel"])
        episode.custom_metrics["avg_co2_emissions"] = np.mean(episode.user_data["avg_co2_emissions"])
        episode.custom_metrics["avg_co_emissions"] = np.mean(episode.user_data["avg_co_emissions"])
        episode.custom_metrics["avg_hc_emissions"] = np.mean(episode.user_data["avg_hc_emissions"])
        episode.custom_metrics["avg_nox_emissions"] = np.mean(episode.user_data["avg_nox_emissions"])

        # worker.env.monitor.evaluate(worker.env, save_traj=False)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--num-iters", type=int, default=2000)
#     args = parser.parse_args()

#     ray.init()
#     trials = tune.run(
#         "PG",
#         stop={
#             "training_iteration": args.num_iters,
#         },
#         config={
#             "env": "CartPole-v0",
#             "callbacks": MyCallbacks,
#         },
#         return_trials=True)

#     # verify custom metrics for integration tests
#     custom_metrics = trials[0].last_result["custom_metrics"]
#     print(custom_metrics)
#     assert "pole_angle_mean" in custom_metrics
#     assert "pole_angle_min" in custom_metrics
#     assert "pole_angle_max" in custom_metrics
#     assert "num_batches_mean" in custom_metrics
#     assert "callback_ok" in trials[0].last_result