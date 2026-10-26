import argparse

import numpy as np
import torch

from dodge.expert import expert_move
from dodge.game import BLUE, MAX_STEPS, RED, TEMPLATES, Game, instruction, other
from dodge.vla import DEVICE, TinyVLA, act


def expert_policy(game, text):
    return expert_move(game.blocks, game.robot, game.bad)  # reads the real rule, ignores the text


def random_policy(seed):
    rng = np.random.default_rng(seed)
    return lambda game, text: int(rng.integers(3))


def vla_policy(model):
    return lambda game, text: act(model, game.grid(), text)


def load_vla(path):
    model = TinyVLA().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model


def told(game, swap, rng):
    """The instruction the policy gets: a random wording of the real rule, or of its opposite."""
    return instruction(other(game.bad) if swap else game.bad, int(rng.integers(len(TEMPLATES))))


def evaluate(policy, episodes, seed, swap=False):
    """Half the episodes use red as the bad color, half blue. With swap=True the policy is told
    the opposite color while the game still scores the real one."""
    rng = np.random.default_rng(seed)
    survived = caught = 0
    for e in range(episodes):
        game = Game(bad=(RED, BLUE)[e % 2], seed=int(rng.integers(2**31)))
        text = told(game, swap, rng)
        while not game.over:
            game.step(policy(game, text))
        survived += not game.crashed
        caught += game.caught
    return {"survived": survived / episodes, "caught": caught / episodes}
