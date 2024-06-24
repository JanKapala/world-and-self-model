from world_and_self_model.hogwild.runners import EnvironmentRunner, AgentRunner, Buffer


# TODO: optional agent and env synchronization
# Universe = Env + Agent(s)


def simulate(
    env: EnvironmentRunner,
    agent: AgentRunner,
    episodes: int,
    max_episode_steps: int,
    render: bool = True,
    seed: int | None = None,
    episodes_pb: bool = True,
    steps_pb: bool = True,
) -> None:
    """Perform simulation of the interactions between the environment and the
    agent.

    Args:
        env: Environment.
        agent: Agent.
        episodes: Number of episodes to play.
        max_episode_steps: Max number of steps in the episode.
        render: Flag, if true then episodes will be rendered.
        seed: Random seed.
        episodes_pb: Flag whether to display episodes progress bar.
        steps_pb: Flag whether to display steps progress bar.

    Returns:

    """

    observation_buffer = Buffer()
    action_buffer = Buffer()
    reward_buffer = Buffer()

    # Connect agent and env together
    agent.reward_buffer = env.reward_buffer
    agent.observation_buffer = env.observation_buffer
    env.action_buffer = agent.action_buffer

    env.start()
    agent.start()