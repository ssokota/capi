"""Policy-value network."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .type_checking import assert_shape


class NN(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, num_items: int, num_utterances: int
    ):
        """Policy-value network

        Semantically distinct actions are given distinct heads.

        Args:
            input_size: Dimension of belief states
            hidden_size: Dimension of hidden layers
            num_items: Number of items in game
            num_utterances: Number of utteraces in game
        """
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fcv = nn.Linear(hidden_size, 1)
        self.fcp1 = nn.Linear(hidden_size, num_items * num_utterances)
        self.fcp2 = nn.Linear(hidden_size, num_items * num_utterances)
        self.fctrade = nn.Linear(hidden_size, 2 * num_items ** 3)
        self.num_utterances = num_utterances
        self.num_items = num_items

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.tensor, Tuple[torch.tensor, torch.tensor, torch.tensor]]:
        """Do forward pass

        Args:
            x [*, 2 num_items + 2 num_utterances]: Tensor representation
                of public belief state

        Returns [*, 1]: values,
                (
                    [*, num_items, num_utterances]: player 1 utterance logits,
                    [*, num_items, num_utterances]: player 2 utterance logits,
                    [*, 2 num_items, num_items^2]: trade logits
                )
        """
        assert_shape(x, (2 * self.num_items + 2 * self.num_utterances,), dim=-1)
        batch_shape = x.shape[:-1]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x) + x)
        x = F.relu(self.fc3(x) + x)
        v = self.fcv(x)
        p1 = self.fcp1(x).view(*batch_shape, self.num_items, self.num_utterances)
        p2 = self.fcp2(x).view(*batch_shape, self.num_items, self.num_utterances)
        trade = self.fctrade(x).view(
            *batch_shape, 2 * self.num_items, self.num_items ** 2
        )
        return v, (p1, p2, trade)
