import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dodge.expert import expert_move
from dodge.game import BLUE, RED, TEMPLATES, Game, instruction, render
from dodge.vla import DEVICE, TinyVLA, tokenize


def collect(episodes, seed):
    """Play expert games and label every frame twice, once per task.

    Each frame shows up with both instructions and their different correct moves, so the frame
    alone cannot tell the model which color to dodge; only the text can.
    Returns grids (N, ROWS, COLS), words (N, MAX_WORDS), moves (N,), bads (N,).
    """
    rng = np.random.default_rng(seed)
    grids, words, moves, bads = [], [], [], []
    for _ in range(episodes):
        game = Game(bad=int(rng.choice([RED, BLUE])), seed=int(rng.integers(2**31)))
        while not game.over:
            for bad in (RED, BLUE):
                grids.append(game.grid())
                words.append(tokenize(instruction(bad, int(rng.integers(len(TEMPLATES))))))
                moves.append(expert_move(game.blocks, game.robot, bad))
                bads.append(bad)
            game.step(expert_move(game.blocks, game.robot, game.bad))
    return np.array(grids), np.array(words), np.array(moves), np.array(bads)
