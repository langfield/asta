#!/usr/bin/env python
# -*- coding: utf-8 -*-
# type: ignore
""" An example trainer for a simply policy gradient implementation. """
import gym
import torch
from oxentiel import Oxentiel
from torch.optim import Adam

from asta import dims, shapes
from asta.tests.rl.pg import Policy, Trajectories, get_action, compute_loss


def train(ox: Oxentiel) -> None:
    """ Training loop. """

    env: gym.Env = gym.make(ox.env_name)

    shapes.OB = env.observation_space.shape
    dims.NUM_ACTIONS = env.action_space.n

    policy = Policy(shapes.OB[0], dims.NUM_ACTIONS, ox.hidden_dim)
    optimizer = Adam(policy.parameters(), lr=ox.lr)
    trajectories = Trajectories()

    ob = env.reset()
    done = False

    for i in range(ox.iterations):

        # Critical: to add prev ob to trajectories buffer.
        prev_ob = ob

        ob_t = torch.Tensor(ob)
        act = get_action(policy, ob_t)
        ob, rew, done, _ = env.step(act)

        trajectories.add(prev_ob, act, rew)

        if done or (i > 0 and i % ox.batch_size == 0):
            trajectories.finish()
            ob, done = env.reset(), False

        if i > 0 and i % ox.batch_size == 0:
            mean_ret, mean_len = trajectories.stats()
            obs, acts, weights = trajectories.get()

            optimizer.zero_grad()
            batch_loss = compute_loss(policy, obs, acts, weights)
            batch_loss.backward()
            optimizer.step()

            print(
                "Iteration: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f"
                % (i, batch_loss, mean_ret, mean_len)
            )

            del obs
            del acts
            del weights
            del mean_ret
            del mean_len


def test_main() -> None:
    """ Run the trainer. """
    settings = {
        "env_name": "CartPole-v0",
        "lr": 1e-2,
        "hidden_dim": 32,
        "iterations": 500,
        "batch_size": 50,
        "epochs": 50,
    }
    ox = Oxentiel(settings)
    train(ox)
