from collections import deque
from typing import Deque, Type

from gymnasium import Space

from agi.model import Transformer
from agi.hogwild.runners import ReadOnlyBufferProxy, Buffer


class EnvironmentObservation:
    def __init__(self, observation, reward):
        self.observation = observation
        self.reward = reward


class SelfObservation:
    def __init__(self, action, thought):
        self.action = action
        self.thought = thought


class PerceptionFrame:
    def __init__(self, env_obs, self_obs):
        self.env_obs = env_obs
        self.self_obs = self_obs


# TODO: multiepisode learning

class ConsciousAgent:
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        thought_size,
        observation_buffer: ReadOnlyBufferProxy,
        reward_buffer: ReadOnlyBufferProxy,
        action_buffer: Buffer,
        model_class: Type = Transformer,
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        self.memory: Deque[PerceptionFrame] = deque(
            maxlen=time_perception_window_size
        )
        self.memory.append(
            PerceptionFrame(
                env_obs=EnvironmentObservation(
                    observation=None,
                    reward=0,
                ),
                self_obs=SelfObservation(
                    action=None,
                    thought=None,
                ),
            )
        )
        self.model = model

        self._observation_buffer = observation_buffer
        self._reward_buffer = reward_buffer
        self._action_buffer = action_buffer
        self._thought = ?
        self._randomness_source

        self._total_return = 0

    # TODO: restart of training: what should be value of the random seed - rather not like when the training begun.

    def percept_and_act_step(self):
        #It is possible to desynchronie all of it, but for now let's KISS

        # OTAR: from buffers to tensors
        O = tensorize(self._observation_buffer.value)
        T = tensorize(self._thought_buffer.value)
        A = tensorize(self._action_buffer.value)
        R = tensorize(self._reward_buffer.value)

        self._total_return += R

        frame = to_frame_tensor(O, T, A, R)
        self.memory = torch.stack(self.memory, frame)[-max_memory_size:]

        seq = self.memory[-seq_len:]
        next_frame = self.model(seq)[-1]
        _, T, A, _ = from_frame_tensor(next_frame)

        # TA: from buffers to tensors
        self._action_buffer.value = untensorize(A)
        self._thought = T


    def learn_step(
        self,

    ):
        x = self.memory[:seq_len]
        y = self.memory[-seq_len:]

        y_pred = self.model(x)

        l = loss(y, y_pred, self._total_return)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
