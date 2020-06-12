#!/usr/bin/env python
# -*- coding: utf-8 -*-
# type: ignore
""" Functions for a simple policy gradient. """
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from asta import Array, Tensor, dims, shapes, symbols, typechecked

X = symbols.X
OB = shapes.OB
N_ACTS = dims.NUM_ACTIONS


@typechecked
def get_policy_distribution(
    policy: nn.Module, obs: Tensor[float, (*OB,)]
) -> Categorical:
    """ Computes the policy distribution for the state given by ``ob``. """
    logits = policy(obs)
    return Categorical(logits=logits)


@typechecked
def get_batch_policy_distribution(
    policy: nn.Module, obs: Tensor[float, (-1, *OB)]
) -> Categorical:
    """ Computes the policy distribution for a batch of observations. """
    logits = policy(obs)
    return Categorical(logits=logits)


@typechecked
def get_action(policy: nn.Module, obs: Tensor[float, (*OB,)]) -> int:
    """ Sample action from policy. """
    distribution = get_policy_distribution(policy, obs)
    sample: Tensor[int, ()] = distribution.sample()
    act: int = sample.item()
    return act


@typechecked
def compute_loss(
    policy: nn.Module,
    obs: Tensor[float, (X, *OB)],
    act: Tensor[int, X],
    weights: Tensor[float, X],
) -> Tensor[float, ()]:
    """ See ``LOSS.txt``. """
    logp = get_batch_policy_distribution(policy, obs).log_prob(act)
    return -(logp * weights).mean()


class Policy(nn.Module):
    """ The parameterized action-value and state-value functions. """

    def __init__(self, observation_size: int, num_actions: int, hidden_dim: int):
        super().__init__()
        self.num_actions: int = num_actions
        self._policy = nn.Sequential(
            nn.Linear(observation_size, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_actions),
            nn.Identity(),
        )

    @typechecked
    def forward(self, ob: Tensor[float, (..., *OB)]) -> Tensor[float, ..., N_ACTS]:
        r""" Actor which implements the policy $\pi_{\theta}$. """
        logits = self._policy(ob)
        return logits


class Trajectories:
    """ An object to hold trajectories. """

    def __init__(self) -> None:
        self.obs: List[Array[float, OB]] = []
        self.acts: List[int] = []
        self.rews: List[float] = []
        self.weights: List[float] = []
        self.ep_rets: List[float] = []
        self.ep_lens: List[int] = []

        self.rets: List[float] = []
        self.lens: List[int] = []

    def add(self, ob: Array[float, OB], act: int, rew: float) -> None:
        """ Add an observation, action, and reward to storage. """
        self.obs.append(ob)
        self.acts.append(act)
        self.rews.append(rew)

    @typechecked
    def get(
        self,
    ) -> Tuple[
        Tensor[float, (X, *OB)], Tensor[int, X], Tensor[float, X],
    ]:
        """ Return observations, actions, and weights for a batch. """
        assert len(self.obs) == len(self.acts) == len(self.weights)

        # Cast buffer storage to tensors.
        obs_t = torch.Tensor(self.obs)
        acts_t = torch.Tensor(self.acts).int()
        weights_t = torch.Tensor(self.weights)

        # Reset.
        self.obs = []
        self.acts = []
        self.weights = []
        self.rets = []
        self.lens = []

        return obs_t, acts_t, weights_t

    def stats(self) -> Tuple[float, float]:
        """ Return the current mean episode return and length. """
        mean_ret = np.mean(self.rets)
        mean_len = np.mean(self.lens)
        return mean_ret, mean_len

    def finish(self) -> None:
        """ Compute and save weights for a (possibly partially completed) episode. """
        ep_ret = sum(self.rews)
        ep_len = len(self.rews)
        self.rets.append(ep_ret)
        self.lens.append(ep_len)
        self.weights.extend([ep_ret] * ep_len)

        assert len(self.obs) == len(self.acts) == len(self.weights)

        # Reset.
        self.rews = []
