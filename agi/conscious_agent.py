import random

import torch
import torch.nn.functional as F
from gymnasium.spaces import flatdim, flatten, unflatten, Box
from mlflow import log_metric
from numpy._typing import NDArray
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from agi.memory import Memory
from agi.model import Transformer

# TODO: multiepisode learning
# TODO: randomness out of the loss function
# TODO: 2D Attention


# TODO: check if splitting is ok, maybe view wouldbe better (what about gradient, etc.)


class ConsciousAgent:
    def __init__(
            self,
            observation_space: Box,
            action_space: Box,
            d_model: int,
            working_memory_size: int,
            short_term_memory_size: int,
            thought_size,
            randomness_source_size: int,
            batch_size: int,
            heads_n: int,
            blocks_n: int,
            dropout: int,
            lr: float,
            gradient_max: float,
            device: str = "cuda",
            writer: SummaryWriter | None = None,
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        self.memory = None
        self.short_term_memory_size = short_term_memory_size
        self.working_memory_size = working_memory_size

        self.heads_n = heads_n
        self.blocks_n = blocks_n
        self.dropout = dropout

        self.d_model = d_model
        self.model: Transformer | None = None

        self.thought_size = thought_size
        self._randomness_source_size = randomness_source_size

        self.device = device

        self._last_thought_tensor = None
        self._last_action_tensor = None
        self._return = None
        self.reset()

        self.batch_size = batch_size
        self._split_sizes = None

        self._optimizer = None
        self.lr = lr
        self.epsilon = 1.0

        self._min_total_return = torch.inf
        self._max_total_return = -torch.inf

        self.writer = writer

        self._global_step = 0
        self._gradient_max = gradient_max

    def register_hooks(self):
        pass
        # for layer in self.model.modules():
        #     layer.register_forward_hook(self.hook_fn)

    # def hook_fn(self, _model, _input, output):
    #     if isinstance(output, tuple):
    #         output = torch.cat(output, dim=-1)
    #     min_val = output.min().item()
    #     max_val = output.max().item()
    #     self._log_metric("Min activation", min_val)
    #     self._log_metric("Max activation", max_val)

    def reset(self) -> None:
        if self._return is not None:
            self._log_metric("return", self._return.item())
        if self.model is not None:
            self._log_histograms()
        self._last_thought_tensor = torch.zeros(self.thought_size)
        self._last_action_tensor = torch.zeros(flatdim(self.action_space))
        self._return = torch.tensor([float(0)])

    # TODO: restart of training: what should be value of the random seed - rather not like when the training begun.

    def step(self, obs: NDArray, reward: float) -> NDArray:
        observation_tensor = self._obs2tensor(obs)
        reward_tensor = self._reward2tensor(reward)
        random_tensor = self._get_scaled_random_tensor()

        self._return += reward_tensor  # TODO: total return scaling?

        thought_tensor, action_tensor = self._step(
            observation=observation_tensor,
            thought=self._last_thought_tensor,
            randomness=random_tensor,
            action=self._last_action_tensor,
            reward=reward_tensor,
            total_return=self._return,
        )

        self._last_thought_tensor = thought_tensor
        self._last_action_tensor = action_tensor

        action = self._action_tensor2action(action_tensor)

        self._global_step += 1

        return action

    def _obs2tensor(self, observation: NDArray) -> Tensor:
        return self._scale(
            x=torch.tensor(flatten(self.observation_space, observation)),
            old_low=self.observation_space.low,
            old_high=self.observation_space.high,
            new_low=-1,
            new_high=1,
        )

    def _reward2tensor(self, reward: float) -> Tensor:
        # TODO: scale reward with moving average
        return torch.tensor([float(reward)])

    def _get_scaled_random_tensor(self, batch_size: int | None = None, seq_len: int | None = None):
        dims = [batch_size, seq_len, self._randomness_source_size]
        shape = [dim for dim in dims if dim is not None]
        return self._scale(
            x=torch.rand(*shape),
            old_low=0,
            old_high=1,
            new_low=-1,
            new_high=1,
        )

    def _action_tensor2action(self, action_tensor: Tensor) -> NDArray:
        return unflatten(self.action_space, self._scale(
            x=action_tensor.cpu().numpy(),
            old_low=-1,
            old_high=1,
            new_low=self.action_space.low,
            new_high=self.action_space.high
        ))

    def _step(
            self,
            observation: Tensor,
            thought: Tensor,
            randomness: Tensor,
            action: Tensor,
            reward: Tensor,
            total_return: Tensor,
    ) -> tuple[Tensor, Tensor]:
        perception_frame_parts = (
            observation,
            thought,
            randomness,
            action,
            reward,
            total_return
        )

        self._lazy_init(perception_frame_parts)
        self.memory.append(torch.cat(perception_frame_parts))
        # TODO: learning steps per env step Hparam
        self._learn_step()
        wm = self.memory.working_memory
        thought, action = self._next_thought_and_action(wm)
        return thought, action

    def _lazy_init(self, perception_frame_parts) -> None:
        self._split_sizes = self._split_sizes or [
            len(fp) for fp in perception_frame_parts
        ]
        perception_frame_size = sum(self._split_sizes)
        self.memory = self.memory or Memory(
            short_term_memory_size=self.short_term_memory_size,
            working_memory_size=self.working_memory_size,
            perception_frame_size=perception_frame_size,
        )

        self.model = self.model or Transformer(
            perception_frame_size=perception_frame_size,
            d_model=self.d_model,
            seq_len=self.working_memory_size,
            heads_n=self.heads_n,
            blocks_n=self.blocks_n,
            dropout=self.dropout,
        ).to(self.device)
        self.register_hooks()

        self._optimizer = self._optimizer or torch.optim.Adam(
            self.model.parameters(), lr=self.lr)



    def _next_thought_and_action(
        self,
        working_memory_content: Tensor
    ) -> tuple[Tensor, Tensor]:
        next_perception_frame_logits, _ = self.model.next_frame(
            working_memory_content.unsqueeze(dim=0).to(self.device)
        )
        next_perception_frame_logits = next_perception_frame_logits.squeeze().cpu()
        _, thought_logits, _, action_logits, _, _ = torch.split(
            next_perception_frame_logits, self._split_sizes
        )

        self._log_metric("epsilon", self.epsilon)
        if random.random() < self.epsilon:
            action = self._scale(
                x=torch.rand_like(action_logits),
                old_low=0,
                old_high=1,
                new_low=-1,
                new_high=1
            )
        else:
            action = F.tanh(action_logits)
        thought = F.tanh(thought_logits)

        return thought, action

    @staticmethod
    def _scale(x, old_low, old_high, new_low, new_high):
        x = (x - old_low) / (old_high - old_low)
        x = new_low + x * (new_high - new_low)
        return x

    def _learn_step(self) -> None:
        batch = self.memory.get_batch(batch_size=self.batch_size)
        features = batch[:, :self.working_memory_size, :].to(self.device)
        targets = batch[:, -self.working_memory_size:, :].to(self.device)
        logits, features_expected_returns = self.model(features)
        loss = self._loss_function(logits, targets, features_expected_returns, self._split_sizes)
        self._optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=self._gradient_max)
        self._optimizer.step()

    def _log_metric(self, key: str, value: float) -> None:
        log_metric(key, value, step=self._global_step)
        self.writer.add_scalar(key, value,
                               global_step=self._global_step)

    def _loss_function(self, logits: Tensor, targets: Tensor, current_expected_returns: Tensor, split_sizes: list[int]) -> Tensor:
        # wm = working memory
        predicted_next_wm = torch.split(logits,  split_sizes, dim=-1)
        target_next_wm = torch.split(targets, split_sizes, dim=-1)
        obs_logits,  thoughts_logits,  random_tensors, actions_logits,  rewards_preds,   total_return_preds    = predicted_next_wm
        obs_targets, thoughts_targets, ______________, actions_targets, rewards_targets, returns_targets = target_next_wm

        # Model should progress on predicting next wm (especially obs)
        obs_preds = F.tanh(obs_logits)
        obs_loss = F.mse_loss(obs_preds, obs_targets)

        thoughts_preds = F.tanh(thoughts_logits)
        thoughts_loss = F.mse_loss(thoughts_preds, thoughts_targets)  # TODO: remove?

        actions_preds = F.tanh(actions_logits)
        actions_loss = F.mse_loss(actions_preds, actions_targets)  # TODO: remove?

        rewards_loss = F.mse_loss(rewards_preds, rewards_targets)

        returns_loss = F.mse_loss(total_return_preds, returns_targets)

        # TODO: maybe something else than mean
        total_perception_loss = torch.stack([
            obs_loss,
            # thoughts_loss,
            # actions_loss,
            # rewards_loss,
            # returns_loss
        ]).mean()

        # next_wm is equal to target_next_wm but thoughts and actions (and random) are
        # replaced with predicted ones (and new one). That's because the model
        # has changed from the time when target_next_wm has been created so
        # next thoughts and actions will be different. Of course random tensor
        # also.
        batch_size, seq_len, _ =obs_logits.shape
        random_tensor = self._get_scaled_random_tensor(batch_size=batch_size, seq_len=seq_len).to(self.device)
        next_wm = torch.cat([obs_targets, thoughts_preds, random_tensor, actions_preds, rewards_targets, returns_targets], dim=-1)
        _, next_expected_returns = self.model(next_wm)
        next_rewards = rewards_targets

        expected_return_loss = F.mse_loss(current_expected_returns, next_rewards + next_expected_returns)

        alpha = 0.5  # TODO: make hparam
        total_loss = (1-alpha)*total_perception_loss + alpha*expected_return_loss

        # self._log_metric("obs_loss", obs_loss.item())
        # self._log_metric("thoughts_loss", thoughts_loss.item())
        # self._log_metric("actions_loss", actions_loss.item())
        # self._log_metric("rewards_loss", rewards_loss.item())
        # self._log_metric("returns_loss", returns_loss.item())
        self._log_metric("total_perception_loss", total_perception_loss.item())
        self._log_metric("expected_return_loss", expected_return_loss.item())
        self._log_metric("total_loss", total_loss.item())

        return total_loss

    def _log_histograms(self) -> None:
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(  # type: ignore[no-untyped-call]
                tag=f"{name}/params",
                values=param.data.detach().cpu().float().numpy(),
                global_step=self._global_step,
            )
            if param.requires_grad and param.grad is not None:
                self.writer.add_histogram(  # type: ignore[no-untyped-call]
                    tag=f"{name}/grads",
                    values=param.grad.detach().cpu().float().numpy(),
                    global_step=self._global_step,
                )
