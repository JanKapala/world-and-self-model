import os
from time import sleep
from typing import cast

import gymnasium as gym
import mlflow
from gymnasium.spaces import Box
from mlflow import log_metric
from torch.utils.tensorboard import SummaryWriter

from world_and_self_model.conscious_agent import ConsciousAgent
from world_and_self_model.simulation import simulate
from constants import TENSORBOARD_LOGS_PATH, MLFLOW_BACKEND_STORE_PATH

if __name__ == "__main__":
    experiment_name = "some_training"

    mlflow.set_tracking_uri(MLFLOW_BACKEND_STORE_PATH)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)

    mlflow.enable_system_metrics_logging()

    env = gym.make("LunarLanderContinuous-v2", render_mode=None)


    # TODO:
    # smaller lr
    # gradient clipping
    # layernorm

    writer = SummaryWriter(log_dir=os.path.join(TENSORBOARD_LOGS_PATH, experiment_name))
    agent = ConsciousAgent(
        observation_space=cast(Box, env.observation_space),
        action_space=cast(Box, env.action_space),
        d_model=256,
        working_memory_size=64,
        short_term_memory_size=150,
        thought_size=32,
        randomness_source_size=1,
        batch_size=32,
        heads_n=4,
        blocks_n=2,
        dropout=0,
        gradient_max=1,
        lr=1e-2,
        device="cuda",
        writer=writer
    )

    simulate(
        env=env,
        agent=agent,
        episodes=100,
        max_episode_steps=1000,
        render=False,
        seed=42,
        episodes_pb=True,
        steps_pb=True,
    )

    env = gym.make("LunarLanderContinuous-v2", render_mode="human")
    simulate(
        env=env,
        agent=agent,
        episodes=100,
        max_episode_steps=1000,
        render=True,
        seed=42,
        episodes_pb=True,
        steps_pb=True,
    )
