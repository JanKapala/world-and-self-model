import random
from collections import deque
from typing import TypeVar, List

import torch
from torch import Tensor

T = TypeVar("T")


class SlicedDeque(deque[T]):
    def __getitem__(self, item: int | slice) -> T | List[T]:
        if isinstance(item, slice):
            start, stop, step = item.indices(len(self))
            return SlicedDeque([self[i] for i in range(start, stop, step)])
        else:
            return super().__getitem__(item)


class Memory:
    def __init__(self, short_term_memory_size, working_memory_size,
                 perception_frame_size):
        self.short_term_memory_size = short_term_memory_size
        self.working_memory_size = working_memory_size
        self.perception_frame_size = perception_frame_size
        self.perception_frames: SlicedDeque[Tensor] = SlicedDeque(
            maxlen=short_term_memory_size)

    def append(self, perception_frame: Tensor) -> None:
        self.perception_frames.append(perception_frame)

    @property
    def working_memory(self) -> Tensor:
        wm = torch.zeros(self.working_memory_size, self.perception_frame_size)

        frames = self.perception_frames[-self.working_memory_size:]
        if not frames:
            return wm

        frames = torch.stack(list(frames), dim=0)
        wm[-len(frames):] = frames
        return wm

    def get_batch(self, batch_size: int) -> Tensor:
        frames = self.perception_frames
        frame_size = self.perception_frame_size
        wm_size = self.working_memory_size

        padded_batch = torch.zeros(batch_size, wm_size + 1, frame_size)
        max_idx = len(frames) - (wm_size + 1)
        if max_idx < batch_size:
            return padded_batch

        start_indices = random.sample(
            population=range(max_idx),
            k=batch_size,
            counts=range(1, max_idx+1)
        )
        return torch.stack(
            [
                torch.stack(
                    list(frames[start_idx:start_idx + wm_size + 1]), dim=0
                ) for start_idx in start_indices
            ],
            dim=0
        )



    def __len__(self):
        return len(self.perception_frames)


if __name__ == "__main__":
    short_term_memory_size = 50
    working_memory_size = 7
    perception_frame_size = 2

    batch_size = 5
    m = Memory(short_term_memory_size, working_memory_size, perception_frame_size)

    target_working_memory = torch.zeros(working_memory_size, perception_frame_size)
    target_batch = torch.zeros(batch_size, working_memory_size+1, perception_frame_size)
    assert torch.allclose(m.working_memory, target_working_memory)
    assert torch.allclose(m.get_batch(batch_size), target_batch)

    frames = []
    for i in range(short_term_memory_size+1):
        frame = torch.rand(2)
        m.append(frame)
        frames.append(frame)

        target_working_memory = torch.zeros(working_memory_size, perception_frame_size)
        temp = torch.stack(frames[-working_memory_size:])
        target_working_memory[-temp.shape[0]:] = temp
        print("Good")
        assert torch.allclose(m.working_memory, target_working_memory)

        if len(frames) < batch_size+working_memory_size+1:
            target_batch = torch.zeros(batch_size, working_memory_size + 1, perception_frame_size)
            assert torch.allclose(m.get_batch(batch_size), target_batch)
        else:
            target_batch = torch.zeros(batch_size, working_memory_size + 1, perception_frame_size)
            batch = m.get_batch(batch_size)
            assert not torch.allclose(batch, target_batch)



