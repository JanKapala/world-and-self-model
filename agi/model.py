# pylint: disable=expression-not-assigned, unused-import

"""The Transformer model definition."""

from math import inf, sqrt
from typing import Generator, cast

import torch
import torch.nn.functional as F
from torch import (  # pylint: disable=no-name-in-module
    Tensor,
    arange,
    cat,
    cos,
    multinomial,
    no_grad,
    ones,
    sin,
    tril,
    zeros,
)
from torch.nn import (
    GELU,
    BatchNorm1d,
    Dropout,
    Embedding,
    LayerNorm,
    LeakyReLU,
    Linear,
    Module,
    ReLU,
    Sequential, Sigmoid, Tanh,
)
from torch.nn.init import kaiming_uniform_


class Transformer(Module):
    """Transformer model."""

    def __init__(
        self,
        perception_frame_size: int,
        d_model: int,
        heads_n: int,
        blocks_n: int,
        seq_len: int,
        dropout: float,
    ) -> None:
        """The Transformer model with the architecture parametrized by the
        given values.

        Args:
            d_model: The diameter of the model
            heads_n: Number of the Casual self attention heads in the
                transformer block.
            blocks_n: Number of blocks (layers) of the transformer model.
            seq_len: Max size of the sequence that can fit into this
                transformer. So-called time perception window size. It is also
                the dimension of the attention matrix created during forward
                pass.
            dropout: Percentage of the dropout across the model layers.
        """
        super().__init__()
        # WARNING: this is important for the beginning phase of training
        # because by default embeddings are initialized with empty tensor.
        # TODO: but GELU are used in the nlp and kaiming is for relu.
        self.embedding = Linear(perception_frame_size, d_model)
        self.positional_encoding = PositionalEncoding(seq_len, d_model)
        self.blocks = Sequential(
            *[
                Block(heads_n, d_model, seq_len, dropout)
                for _ in range(blocks_n)
            ],
            LayerNorm(d_model), # TODO: it is not present in the original
            #  nlp paper but it is probably in the karpathy/gpt version
        )
        self.net = Sequential(
            self.embedding,
            self.positional_encoding,
            Dropout(dropout),
            self.blocks,
        )
        self.expected_return_layer = Linear(d_model, 1)

    # pylint: disable-next=missing-function-docstring
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """I predict next working memory based on current and it calculates
        expected returns for current working memory frames.

        Args:
            x: Current working memory tensor.

        Returns:
            Tuple of tensors:
            - The predicted next working memory tensor
            - The expected return for current working memory frames
        """
        logits = self.net(x)
        perception_frame = F.linear(logits, self.embedding.weight.t())
        expected_return = self.expected_return_layer(logits)
        seq_len, _ = self.positional_encoding.encodings.shape
        return perception_frame, expected_return

    def next_frame(self, working_memory: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, input_seq_len, _ = working_memory.shape
        seq_len, _ = self.positional_encoding.encodings.shape
        device = working_memory.device
        perception_frame_size = self.embedding.weight.shape[-1]
        padded_wm = torch.zeros(batch_size, seq_len, perception_frame_size).to(device)
        padded_wm[:, -input_seq_len:, :] = working_memory
        with no_grad():
            next_frame, expected_return = self(padded_wm)
            return next_frame[:, -1, :], expected_return

    def count_parameters(self) -> int:
        """Count all parameters of the model.
        WARNING: it includes also parameters from the LayerNorm, BatchNorm, etc.
        """
        return sum(
            param.numel() for param in self.parameters() if param.requires_grad
        )


class PositionalEncoding(Module):
    """Positional encoding according to the
    `Attention is all you need paper.`"""

    _ENCODINGS_KEY = "_encodings"

    def __init__(self, seq_len: int, d_model: int) -> None:
        super().__init__()
        encodings = self._create_pe(seq_len, d_model)
        self.register_buffer(self._ENCODINGS_KEY, encodings)

    @property
    def encodings(self) -> Tensor:
        """Get encodings tensor."""
        return cast(Tensor, getattr(self, self._ENCODINGS_KEY))

    @staticmethod
    def _create_pe(seq_len: int, model_dimension: int) -> Tensor:
        # TODO: Use exp form to improve numeric stability
        p = arange(0, seq_len).unsqueeze(1)
        i = arange(0, model_dimension, 2)
        term = p / 10000 ** (i / model_dimension)
        positional_encodings = zeros((seq_len, model_dimension))
        positional_encodings[:, 0::2] = sin(term)
        positional_encodings[:, 1::2] = cos(term)
        return positional_encodings

    # pylint: disable-next=missing-function-docstring
    def forward(self, x: Tensor) -> Tensor:
        return x + self.encodings[: x.shape[-2], :]


class Block(Module):
    """Single Transformer Block."""

    def __init__(
        self, heads_n: int, d_model: int, seq_len: int, dropout: float
    ) -> None:
        """Single transformer Block parametrized by the given values.

        Args:
            heads_n: Number of the casual self attention heads in the block.
            d_model: The diameter of the model and also a token embedding
                dimension.
            seq_len: Max size of the sequence that can fit into this
                transformer. So-called context size. It is also the dimension
                of the attention matrix created during forward pass.
            dropout: Percentage of the dropout across the model layers.
        """

        super().__init__()
        self.cmhsa = CasualMultiHeadSelfAttention(d_model, heads_n, seq_len)
        self.dropout_1 = Dropout(dropout)
        self.ln_1 = LayerNorm(d_model)
        self.ff = FeedForward(d_model)
        self.dropout_2 = Dropout(dropout)
        self.ln_2 = LayerNorm(d_model)

    # pylint: disable-next=missing-function-docstring
    def forward(self, x: Tensor) -> Tensor:
        x = self.ln_1(x + self.dropout_1(self.cmhsa(x)))
        x = self.ln_2(x + self.dropout_2(self.ff(x)))
        return x


class CasualMultiHeadSelfAttention(Module):
    """Casual Multi-head Self Attention block implemented according to the
    "Attention is all you need" paper.
    """

    def __init__(self, d_model: int, heads_n: int, seq_len: int) -> None:
        super().__init__()
        self.kvq_projections = Linear(d_model, 3 * d_model)
        self.h_n = heads_n
        self.h_s = d_model // heads_n
        self.register_buffer("mask", tril(ones(seq_len, seq_len)))

    # pylint: disable-next=missing-function-docstring
    def forward(self, x: Tensor) -> Tensor:
        # pylint: disable=invalid-name
        B, T, C = x.shape

        x = self.kvq_projections(x)
        x = x.reshape(B, T, 3, self.h_n, self.h_s)  # B, T, 3, H_n, H_s
        x = x.permute(0, 2, 3, 1, 4)  # B, 3, H_n, T, H_s
        q, k, v = x.unbind(1)  # 3xB, H_n, T, H_s

        A = q @ k.transpose(-1, -2) / sqrt(self.h_s)  # B, h_n, seq, seq
        A = F.softmax(A.masked_fill(self.mask == 0, -inf), dim=-1)
        x = A @ v  # B, H_n, T, H_s

        x = x.transpose(1, 2).reshape(B, T, C)  # B, T, C

        return x


class FeedForward(Module):
    """Feed-Forward block implemented according to the
    "Attention is all you need" paper.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        ff1 = Linear(d_model, 4 * d_model)
        ff2 = Linear(4 * d_model, d_model)
        [kaiming_uniform_(ff.weight, nonlinearity="relu") for ff in (ff1, ff2)]
        self.net = Sequential(ff1, ReLU(), ff2)

    # pylint: disable-next=missing-function-docstring
    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.net(x))
