"""Trade Comm PuB-MDP. Supports arbitrary numbers of items and arbitrary
numbers of utterances.
"""

import torch

from .type_checking import assert_joint_probability, assert_prescription, assert_shape


class Game:
    def __init__(self, num_items: int, num_utterances: int):
        """Trade Comm PuB-MDP

        Args:
            num_items: Number of items in the game
            num_utterances: Number of utterances in the game

        Attributes:
            init_dist (torch.Tensor[num_items, num_items]): Initial joint
                probability distribution over player items
            num_utterances (int): Number of utterances in game
        """
        self.init_dist = torch.ones((num_items, num_items)) / (num_items ** 2)
        self.num_utterances = num_utterances

    def init_state(self) -> "State":
        """Return the initial state of the game"""
        return State(self.init_dist, self.num_utterances, [])


class State:
    def __init__(self, dist: torch.tensor, num_utterances: int, spub: list):
        """Trade Comm public belief state

        Args:
            dist [num_items, num_items]: Joint probability distribution over
                player items
            num_utterances: Number of utterances in game
            spub: Public state (list of public observations)

        Attributes:
            See args
        """
        assert_joint_probability(dist, dist.shape)
        self.dist = dist.clone()
        self.num_utterances = num_utterances
        self.spub = list(spub)

    def next_distribution(
        self, prescription: torch.tensor, action: int
    ) -> torch.tensor:
        """Return next distribution over private information

        Given the current belief, return the next belief given the
        coordinator's prescription (Pub-MDP action) and the player's action
        (PuB-MDP observation). For trade comm, this amounts to zeroing-out
        the probability of items that are inconsistent with the
        prescription-action pair for the acting player and renormalizing.

        Args:
            prescription [num_items, num_utterances]: The coordinator's
                action. Each row should be one-hot.
            action: The action of the Dec-POMDP player

        Returns [num_items, num_items]:
            Updated belief
        """
        assert_prescription(prescription, (self.dist.shape[0], self.num_utterances))
        shape = (-1, 1) if self.time() == 0 else (1, -1)
        scores = self.dist * prescription[:, action].view(*shape)
        next_dists = scores / scores.sum()
        assert_joint_probability(next_dists, next_dists.shape)
        return next_dists

    def apply_action(self, prescription: torch.tensor, action: int) -> None:
        """Update the public belief state

        Args:
            prescription: [num_items, num_utterances]: The coordinator's
                action. Each row should be one hot.
            action: The action of the Dec-POMDP player
        """
        self.dist = self.next_distribution(prescription, action)
        self.spub.append(action)

    def tensor(self) -> torch.tensor:
        """Return the public belief state in tensor form

        Note that what this method returns is NOT actually the public
        belief state but rather information that is bijective with the
        public belief state.

        Returns [2 num_items + 2 num_utterances,]:
            Concatenated (posterior over player 1 item,
                          posterior over player 2 item,
                          player 1's utterance,
                          player 2's utterance)
        """
        p1 = self.dist.sum(axis=-1)
        p2 = self.dist.sum(axis=-2)
        a = torch.zeros(2 * self.num_utterances)
        if len(self.spub) > 0:
            a[self.spub[0]] = 1
        if len(self.spub) > 1:
            a[self.num_utterances + self.spub[1]] = 1
        x = torch.cat([p1, p2, a])
        assert_shape(x, (2 * self.dist.shape[0] + 2 * self.num_utterances,))
        return x

    def clone(self) -> "State":
        """Return a deepcopy of the State instance"""
        return State(self.dist.clone(), self.num_utterances, self.spub)

    def time(self) -> int:
        """Return the time step"""
        return len(self.spub)
