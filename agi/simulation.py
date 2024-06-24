import numpy as np
from gymnasium import Env
from tqdm import trange, tqdm

from agi.conscious_agent import ConsciousAgent


def simulate(
    env: Env,
    agent: ConsciousAgent,  # TODO: create an abstract agent class
    episodes: int,
    max_episode_steps: int,
    render: bool = True,
    seed: int | None = None,
    episodes_pb=True,  # TODO: maybe remove this argument.
    steps_pb=True,  # TODO: maybe remove this argument.
) -> (
    None
):  # TODO: Maybe simulation should return some results that could be analysed/saved.
    """Simulate dynamics between an Agent and an Environment.

    :param env: Environment.
    :param agent: Reinforcement Learning Agent.
    :param episodes: Number of episodes to play.
    :param max_episode_steps: Max number of steps in the episode.
    :param render: Flag, if true then episodes will be rendered.
    :param seed: PRNG seed.
    :param episodes_pb:
    :param steps_pb:
    :return:
    """


    start_epsilon = 1.0
    end_epsilon = 0.2
    epsilon = np.linspace(start_epsilon, end_epsilon, episodes)

    for i in trange(episodes, position=0, leave=True, disable=(not episodes_pb)):
        obs, _ = env.reset(seed=seed)
        agent.reset()
        episode_steps = 0
        reward = 0

        agent.epsilon = epsilon[i]

        with tqdm(
            total=max_episode_steps, position=1, leave=False, disable=(not steps_pb)
        ) as pbar:
            while True:
                action = agent.step(obs, reward)
                next_obs, reward, terminated, _, _ = env.step(action)

                if episode_steps >= max_episode_steps:
                    terminated = True

                obs = next_obs
                episode_steps += 1
                pbar.update(1)

                if terminated:
                    break
    env.close()