from functools import partial
from threading import Event, Thread

from gymnasium import Env
from tqdm import trange, tqdm


class Buffer:
    # Lock probably not needed
    def __init__(self, value=None):
        self.value = value

    def get_read_only_proxy(self):
        return ReadOnlyBufferProxy(self)


class ReadOnlyBufferProxy:
    def __init__(self, buffer: Buffer):
        self._buffer = buffer

    @property
    def value(self):
        return self._buffer.value


class EnvironmentRunner:
    def __init__(self, env: Env, seed) -> None:
        self._env = env
        self._seed = seed

        self._observation_buffer = Buffer(env.reset(seed=seed)[0])
        self._read_only_observation_buffer_proxy = self._observation_buffer.get_read_only_proxy()

        self._reward_buffer = Buffer(0)
        self._read_only_reward_buffer_proxy = self._reward_buffer.get_read_only_proxy()

        self._read_only_action_buffer_proxy = None

        self._terminated_event = Event()
        self._thread = None

    @property
    def observation_buffer(self) -> ReadOnlyBufferProxy:
        if self._read_only_observation_buffer_proxy.value is None:
            raise ValueError("`.observation_buffer can be accessed only after "
                             "internal environment's initialization of the "
                             "self._observation_buffer.value`")
        return self._read_only_observation_buffer_proxy

    @property
    def reward_buffer(self) -> ReadOnlyBufferProxy:
        if self._read_only_reward_buffer_proxy.value is None:
            raise ValueError("`.reward_buffer can be accessed only after "
                             "internal environment's initialization of the "
                             "self._reward_buffer.value`")
        return self._read_only_reward_buffer_proxy

    @property
    def action_buffer(self) -> None:
        raise AttributeError("`.action_buffer` can only be set, not retrieved")

    @action_buffer.setter
    def action_buffer(self, value: ReadOnlyBufferProxy) -> None:
        if not isinstance(value, ReadOnlyBufferProxy):
            raise ValueError("`.action_buffer` value should be of type "
                             "ReadOnlyBufferProxy and it is meant to be get "
                             "from the agent.")
        self._read_only_reward_buffer_proxy = value

    @property
    def terminated_event(self) -> Event:
        return self._terminated_event

    def start(
        self,
        episodes,
        max_episode_steps,
        render,
        episodes_pb,
        steps_pb,
        clock_hz: int | None = None,
    ) -> None:
        """Just run an environment

        Args:
            episodes: Number of episodes to play.
            max_episode_steps: Max number of steps in the episode.
            render: Flag, if true then episodes will be rendered.
            seed: Random seed.
            episodes_pb: Flag whether to display episodes progress bar.
            steps_pb: Flag whether to display steps progress bar.

        Returns:

        """

        self.join()
        self._thread = Thread(target=partial(
            self._start,
            episodes=episodes,
            max_episode_steps=max_episode_steps,
            render=render,
            episodes_pb=episodes_pb,
            steps_pb=steps_pb,
            clock_hz=clock_hz,
        ))

        self._thread.start()

    def _start(
        self,
        episodes,
        max_episode_steps,
        seed: int,
        render,
        episodes_pb,
        steps_pb,
        clock_hz: int | None = None,
    ) -> None:
        if self._read_only_action_buffer_proxy is None:
            raise ValueError("`.action_buffer` is not set!")

        for _ in trange(
            episodes,
            position=0,
            leave=True,
            disable=(not episodes_pb)
        ):
            self._observation_buffer.value = self._env.reset(seed=self._seed)[0]
            self._terminated_event.clear()
            episode_steps = 0
            with tqdm(
                    total=max_episode_steps,
                    position=1,
                    leave=False,
                    disable=(not steps_pb),
            ) as pbar:
                while True:
                    if render:
                        self._env.render()

                    action = self._read_only_action_buffer_proxy.value
                    next_obs, reward, terminated, _, _ = self._env.step(action)
                    self._observation_buffer.value = next_obs
                    self._reward_buffer.value = reward

                    if episode_steps >= max_episode_steps:
                        terminated = True

                    episode_steps += 1
                    pbar.update(1)

                    if terminated:
                        self._terminated_event.set()
                        break

        self._env.close()

    def join(self):
        if self._thread is not None:
            self._thread.join()


class AgentRunner:
    def __init__(self, agent):
        self._agent = agent

        self._action_buffer = Buffer(self._agent.reset())
        self._read_only_action_buffer_proxy = self._action_buffer.get_read_only_proxy()
        self._read_only_observation_buffer_proxy = None
        self._read_only_reward_buffer_proxy = None

        self._thread = None

    @property
    def action_buffer(self) -> ReadOnlyBufferProxy:
        if self._read_only_action_buffer_proxy.value is None:
            raise ValueError("`.action_buffer can be accessed only after internal agent's initialization of the self._action_buffer.value`")
        return self._read_only_action_buffer_proxy

    @property
    def observation_buffer(self) -> None:
        raise AttributeError("`.observation_handler` can only be set, not retrieved")

    @observation_buffer.setter
    def observation_buffer(self, value: ReadOnlyBufferProxy) -> None:
        if not isinstance(value, ReadOnlyBufferProxy):
            raise ValueError("`.observation_buffer` value should be of type "
                             "ReadOnlyBufferProxy and it is meant to be get "
                             "from the environment.")
        self._read_only_observation_buffer_proxy = value

    @property
    def reward_buffer(self) -> None:
        raise AttributeError("`.reward_buffer` can only be set, not retrieved")

    @reward_buffer.setter
    def reward_buffer(self, value: ReadOnlyBufferProxy) -> None:
        if not isinstance(value, ReadOnlyBufferProxy):
            raise ValueError("`.reward_buffer` value should be of type "
                             "ReadOnlyBufferProxy and it is meant to be get "
                             "from the environment.")
        self._read_only_reward_buffer_proxy = value

    def start(self) -> None:
        """Just run an agent in a separate thread."""

        self.join()
        self._thread = Thread(target=self._start)
        self._thread.start()

    def _start(self):
        if self._read_only_observation_buffer_proxy is None:
            raise ValueError("Observation handler is not set!")
        if self._read_only_reward_buffer_proxy is None:
            raise ValueError("Reward handler is not set!")

        while True:
            self._agent.step(
                observation_buffer=self._read_only_observation_buffer_proxy,
                reward_buffer=self._read_only_reward_buffer_proxy,
                action_buffer=self._action_buffer,
            )

    def join(self):
        if self._thread is not None:
            self._thread.join()
