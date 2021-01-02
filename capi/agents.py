""""PuB-MDP coordinator"""

from copy import deepcopy
from collections import namedtuple
from typing import Tuple

import torch
import torch.nn.functional as F

from .models import NN
from .games import State
from .type_checking import (
    assert_element,
    assert_joint_probability,
    assert_label_prescription,
    assert_num_dims,
    assert_prescription,
    assert_shape,
)


named_ex = namedtuple("named_ex", ["state", "value", "policy"])
GAME_LEN = 3


class Agent:
    def __init__(
        self,
        num_items: int,
        num_utterances: int,
        nn: NN,
        opt: torch.optim.Optimizer,
        num_samples: int,
        epsilon: float,
        policy_weight: float,
        device: torch.device,
    ):
        """Agent operating within PuB-MDP

        Args:
            num_items: Number of items in game
            num_utterances: Number of utterances in game
            nn: Policy-value networks
            opt: Optimizer for `nn`
            num_samples: Number of search rollouts to do
            epsilon: Exploration probability
            policy_weight: Value by which policy loss is scaled
            device: Device on which to train `nn`

        Attributes:
            See args
            buffers (tuple): State, value policy examples
            pub1 (torch.Tensor)
                [num_samples, num_utterances, 2 * num_utterances]:
                Tensor representations of first public states
            pub2 (Tuple[torch.Tensor, ...])
                ([num_samples, num_utterances, 2 * num_utterances], ...):
                Tensor Representations of second public states
            null_action_dynamics (torch.Tensor): Tensor of zeros to pass to
                trainer during terminal transition.
        """
        self.num_items = num_items
        self.num_utterances = num_utterances
        self.nn = nn.to(device)
        self.opt = opt
        self.num_samples = num_samples
        self.epsilon = epsilon
        self.policy_weight = policy_weight
        self.device = device
        self.buffers = tuple(named_ex([], [], []) for _ in range(GAME_LEN))
        self.init_action_repr()
        self.null_action_dynamics = torch.zeros((num_samples, num_utterances))

    def act(
        self, s: State, train: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, float, bool]:
        """Select a prescription for state `s`

        Args:
            s: The public belief state
            train: Whether agent is training or being evaluated

        Returns:
            [num_items, num_utterances] or [2 num_items, num_items ^2]:
                Prescription selected by coordinator
            [num_utterances]: Probability of each public observation
                given `s` and the selected prescription;
            []: the assessed value of the prescription given `s`;
            whether the game is over
        """
        # Compute policy from public belief state. Another setup that
        # may work better in some circumstances is to parameterize the
        # policy as a tabular function of the public state.
        self.nn.eval()
        with torch.no_grad():
            val, all_logits = self.nn(s.tensor().to(self.device))
        # Optional additional step (not taken here): Modify policy by
        # increasing the entropy of some rows to encourage exploration
        policy = F.softmax(all_logits[s.time()], dim=-1)
        # Sample `num_samples` prescriptions from `policy`. Another setup
        # that may work better in some circumstances is to rollout with the
        # `num_samples`-most likely prescriptions (rather than sampling them).
        samples = self.one_hot(s, self.sample(policy)).to(self.device)
        # There is no reward and no possibility of termination in Trade Comm
        # when the time step index is zero or one.
        if s.time() in (0, 1):
            # Public observation probabilities
            action_dynamics = self.action_dynamics(s, samples)
            # Induced distributions over private information
            next_dists = self.next_distributions(s, samples)
            # Next public belief states in tensor repr
            x = self.make_tensor(s, next_dists)
            with torch.no_grad():
                # Evaluate public belief states
                action_vals, _ = self.nn(x)
            # Weight evaluation over next PBS by transition probabilities
            vals = self.average_over_actions(action_vals, action_dynamics)
        # Termination is guaranteed when the time step index is two
        else:
            # Separate prescription vectors into prescriptions
            p1_samples, p2_samples = self.reshape_samples(samples)
            # Expected terminal values for each prescription vector
            vals = self.compute_terminal_values(s, p1_samples, p2_samples)
            # Dummy dynamics for trainer
            action_dynamics = self.null_action_dynamics
        # Index of best prescription (vector)
        idx = torch.multinomial(torch.isclose(vals, vals.max()).float(), 1).item()
        if train:
            # Add data to buffer
            self.add_to_buffer(s, vals[idx], samples[idx])
            # Pick a random prescription to explore
            if torch.rand(1) < self.epsilon:
                idx = torch.multinomial(torch.ones(self.num_samples), 1).item()
        return_tuple = (
            samples[idx].to("cpu"),
            action_dynamics[idx].to("cpu"),
            vals[idx].item(),
            s.time() == (GAME_LEN - 1),
        )
        if s.time() in (0, 1):
            assert_prescription(return_tuple[0], (self.num_items, self.num_utterances))
        else:
            assert_prescription(
                return_tuple[0], (2 * self.num_items, self.num_items ** 2)
            )
        assert_prescription(
            return_tuple[1], (self.num_utterances,), allow_improper=True, pure=False
        )
        return return_tuple

    def add_to_buffer(
        self, state: State, value: torch.Tensor, policy: torch.Tensor
    ) -> None:
        """Add public belief state & policy/value targets to buffer

        Args:
            state: Public belief state
            value []: Assessed value of `state`
            policy [num_items, num_utterances] or [2 num_items, num_items ^2]:
                Best sampled presription (vector) for `state`
        """
        assert_shape(value, ())
        if state.time() in (0, 1):
            assert_prescription(policy, (self.num_items, self.num_utterances))
        else:
            assert_prescription(policy, (2 * self.num_items, self.num_items ** 2))
        self.buffers[state.time()].state.append(state.tensor().to("cpu"))
        self.buffers[state.time()].value.append(value.to("cpu"))
        self.buffers[state.time()].policy.append(policy.to("cpu"))

    def init_action_repr(self) -> None:
        """Make tensor representations of public states"""
        eye = torch.eye(self.num_utterances)
        ls = [
            eye[i * torch.ones(self.num_samples).long()]
            for i in range(self.num_utterances)
        ]
        actions = torch.stack(ls, dim=1)
        zeros = torch.zeros(
            (self.num_samples, self.num_utterances, self.num_utterances)
        )
        self.spub1 = torch.cat([actions, zeros], dim=-1).to(self.device)
        spub2 = []
        for i in range(self.num_utterances):
            tmp = zeros.clone()
            tmp[:, :, i] = 1
            spub2.append(torch.cat([tmp, actions], dim=-1).to(self.device))
        self.spub2 = tuple(spub2)

    def train(self):
        """Train network w/ MSE and CE and wipe buffer"""
        self.opt.zero_grad()
        self.nn.train()
        loss = 0
        for t in range(GAME_LEN):
            x, v, p = self.get_batch(t)
            v_, logits_ = self.nn(x)
            ell_ = logits_[t]
            value_loss = torch.nn.MSELoss()(v_.flatten(), v.flatten())
            policy_loss = (-p * torch.nn.LogSoftmax(dim=-1)(ell_)).sum(dim=-1).mean()
            loss += value_loss + policy_loss * self.policy_weight
        loss.backward()
        self.opt.step()
        self.buffers = tuple(named_ex([], [], []) for _ in range(GAME_LEN))

    def get_batch(self, t: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the data from the buffer for time `t`

        Args:
            t: The time step for which to get data

        Returns:
            [*, self.num_utterances + self.num_items]: public belief states
            [*]: values
            [*, num_items, num_utterances] or [*, 2 num_items, num_items ^ 2]:
                policies
        """
        assert_element(t, (0, 1, 2))
        x = torch.stack(self.buffers[t].state).to(self.device)
        v = torch.stack(self.buffers[t].value).to(self.device)
        p = torch.stack(self.buffers[t].policy).to(self.device)
        assert_shape(x, (2 * self.num_utterances + 2 * self.num_items,), dim=-1)
        assert_num_dims(v, 1)
        if t in (0, 1):
            assert_prescription(p, (p.shape[0], self.num_items, self.num_utterances))
        else:
            assert_prescription(
                p, (p.shape[0], 2 * self.num_items, self.num_items ** 2)
            )
        return x, v, p

    def sample(self, policy: torch.Tensor) -> torch.Tensor:
        """Sample prescriptions from the policy

        Args:
            policy [num_items, num_utterances] or [2 num_items, num_items^2]:
                Distribution over prescriptions

        Returns [num_samples, policy.shape[0]]: Prescription vectors sampled
            from `policy` represented with labels (not one-hot).
        """
        dist_size = policy.shape[-1]
        shape = (self.num_samples, *policy.shape[:-1])
        flat_policy = policy.view(-1, dist_size)
        samples = (
            torch.multinomial(flat_policy, self.num_samples, replacement=True)
            .permute(1, 0)
            .view(shape)
        )
        assert_label_prescription(
            samples, policy.shape[1], (self.num_samples, policy.shape[0])
        )
        return samples

    def one_hot(self, state: State, label_prescriptions: torch.Tensor) -> torch.Tensor:
        """Convert a prescription written with labels to one-hot repr

        If `time` in (0, 1), label_prescription is shape
        [`num_samples`, `num_items`]
        and return value is shape
        [`num_samples`, `num_items`, `num_utterances`].
        If `time` == 2, label_prescription
        is shape [`num_samples`, 2 `num_items`]
        and return value is shape
        [`num_samples`, 2 `num_items`, `num_items`^2].

        Args:
            state: Public belief state
            label_prescription: Each row (indexed by a sample) maps
                item_idx -> action_idx

        Returns:
            Prescriptions in one-hot representation
        """
        if state.time() in (0, 1):
            assert_label_prescription(
                label_prescriptions,
                self.num_utterances,
                (self.num_samples, self.num_items),
            )
        if state.time() == 2:
            assert_label_prescription(
                label_prescriptions,
                self.num_items ** 2,
                (self.num_samples, 2 * self.num_items),
            )
        shape = label_prescriptions.shape[:-1]
        num_spriv = label_prescriptions.shape[-1]
        num_actions = (
            self.num_utterances if state.time() in (0, 1) else self.num_items ** 2
        )
        eye = torch.eye(num_actions)
        prescriptions = eye[label_prescriptions].view(*shape, num_spriv, num_actions)
        if state.time() in (0, 1):
            assert_prescription(
                prescriptions, (self.num_samples, self.num_items, self.num_utterances)
            )
        if state.time() == 2:
            assert_prescription(
                prescriptions,
                (self.num_samples, 2 * self.num_items, self.num_items ** 2),
            )
        return prescriptions

    def action_dynamics(
        self, state: State, prescriptions: torch.Tensor
    ) -> torch.Tensor:
        """Compute the probabilities of public observations

        For Trade Comm, the public observations are the Dec-POMDP actions.

        Args:
            state: Public belief state
            prescriptions [num_samples, num_items, num_utterances]:
                For each sample, for each item, a one-hot vector

        Returns [num_samples, num_utterances]: Probability of each utterance
            for each sample
        """
        assert_prescription(
            prescriptions, (self.num_samples, self.num_items, self.num_utterances)
        )
        axis = -1 if state.time() == 0 else -2
        action_dynamics = torch.einsum(
            "ijk,j", prescriptions, state.dist.sum(dim=axis).to(self.device)
        )
        assert_prescription(
            action_dynamics, (self.num_samples, self.num_utterances), pure=False
        )
        return action_dynamics

    def next_distributions(
        self, state: State, prescriptions: torch.Tensor
    ) -> torch.Tensor:
        """Compute the next distributions

        Args:
            state: Public belief state
            prescriptions: [num_samples, num_items, num_utterances]:
                For each sample, for each item, a one-hot vector

        Returns [num_samples, num_utterances, num_items, num_items]: For each
            sample, for each utterance, the induced distribution over private
            information. All zeros for utterances with probability zero.
        """
        assert_prescription(
            prescriptions, (self.num_samples, self.num_items, self.num_utterances)
        )
        state_dynamics_string = "ijk,jl->ikjl" if state.time() == 0 else "ijk,lj->iklj"
        score = torch.einsum(
            state_dynamics_string, prescriptions, state.dist.to(self.device)
        )
        norm = score.sum(axis=(2, 3))
        norm[norm == 0] = 1
        next_distributions = score / norm.view(*norm.shape, 1, 1)
        assert_joint_probability(
            next_distributions,
            (self.num_samples, self.num_utterances, self.num_items, self.num_items),
            allow_improper=True,
        )
        return next_distributions

    def make_tensor(self, state: State, dist: torch.Tensor) -> torch.Tensor:
        """Make tensor reprs of next public belief states

        Args:
            state: Public belief state
            dist[num_samples, num_utterances, num_items, num_items]:
                Distributions over private information given public observation
                at next time step.

        Returns
            [num_samples, num_utterances, 2 num_utterances + 2 num_items]:
            Next public belief states
        """
        assert_joint_probability(
            dist,
            (self.num_samples, self.num_utterances, self.num_items, self.num_items),
            allow_improper=True,
        )
        a = self.spub1 if state.time() == 0 else self.spub2[state.spub[0]]
        p1 = dist.sum(axis=-1)  # N x A x Pri
        p2 = dist.sum(axis=-2)
        pbs = torch.cat([p1, p2, a], dim=-1)
        assert_shape(
            pbs,
            (
                self.num_samples,
                self.num_utterances,
                2 * self.num_utterances + 2 * self.num_items,
            ),
        )
        return pbs

    def average_over_actions(
        self, action_vals: torch.Tensor, action_dynamics: torch.Tensor
    ) -> torch.Tensor:
        """Take weighted average over values and probabilities

        The assessed value of a prescription is the weighted average of the
        next belief states it induces, where the weights are determined by
        the public observation probabilities. (Player actions are the public
        observations in Trade Comm.)

        Args:
            action_vals [num_samples, num_utterances, 1]: Value for each
                prescription, for each public observation
            action_dynamics [num_samples, num_utterances]: For each
                prescription, for each public observation, the probability of
                that public observation

        Returns [num_samples,]:
            Expected value of `action_vals` wrt `action_dynamics`
        """
        assert_shape(action_vals, (self.num_samples, self.num_utterances, 1))
        assert_prescription(
            action_dynamics, (self.num_samples, self.num_utterances), pure=False
        )
        ev = (action_vals.view(self.num_samples, -1) * action_dynamics).sum(dim=-1)
        assert_shape(ev, (self.num_samples,))
        return ev

    def reshape_samples(
        self, samples: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Separate prescription vectors into prescriptions for each player

        This function also throws out the prescriptions which necessarily fail
        (those in which a player attempts to trade an item it does not have)

        Args:
            samples [num_samples, 2 * num_items, num_items^2]:
                Prescription vectors

        Returns ([num_samples, num_items, num_items],
                 [num_samples, num_items, num_items]):
                Prescriptions for player 1 and prescriptions for player 2
        """
        assert_prescription(
            samples, (self.num_samples, 2 * self.num_items, self.num_items ** 2)
        )
        samples_ = samples.view(
            self.num_samples, 2 * self.num_items, self.num_items, self.num_items
        )
        samples1 = samples_[
            :, torch.arange(self.num_items), torch.arange(self.num_items)
        ]
        samples2 = samples_[
            :,
            self.num_items + torch.arange(self.num_items),
            torch.arange(self.num_items),
        ]
        assert_prescription(
            samples1,
            (self.num_samples, self.num_items, self.num_items),
            allow_improper=True,
        )
        assert_prescription(
            samples2,
            (self.num_samples, self.num_items, self.num_items),
            allow_improper=True,
        )
        return samples1, samples2

    def compute_terminal_values(
        self,
        state: State,
        p1_prescriptions: torch.Tensor,
        p2_prescriptions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the expected terminal reward for `state` and prescriptions

        Args:
            state: Public belief state
            p1_prescriptions [num_samples, num_items, num_items]:
                Player 1's prescriptions
            p2_prescriptions [num_samples, num_items, num_items]:
                Player 2's prescriptions

        Returns [num_samples,]:
            The expected terminal reward of each prescription
        """
        assert_prescription(
            p1_prescriptions,
            (self.num_samples, self.num_items, self.num_items),
            allow_improper=True,
        )
        assert_prescription(
            p2_prescriptions,
            (self.num_samples, self.num_items, self.num_items),
            allow_improper=True,
        )
        values = (
            p2_prescriptions.permute(0, 2, 1)
            * p1_prescriptions
            * state.dist.to(self.device)
        ).sum(dim=(1, 2))
        assert_shape(values, (self.num_samples,))
        return values

