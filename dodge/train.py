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


def scores(model, grids, words):
    device = next(model.parameters()).device
    return model(torch.as_tensor(render(grids), device=device), torch.as_tensor(words, device=device))


@torch.no_grad()
def accuracy(model, data, batch=1024):
    grids, words, moves, _ = data
    hits = sum(
        (scores(model, grids[i:i + batch], words[i:i + batch]).argmax(1).cpu().numpy() == moves[i:i + batch]).sum()
        for i in range(0, len(moves), batch)
    )
    return hits / len(moves)


def train(model, data, epochs, seed, batch=256, lr=1e-3, val=None):
    grids, words, moves, _ = data
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    device = next(model.parameters()).device
    for epoch in range(epochs):
        order = rng.permutation(len(moves))
        total = 0.0
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            loss = F.cross_entropy(scores(model, grids[idx], words[idx]), torch.as_tensor(moves[idx], device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        line = f"epoch {epoch + 1}/{epochs}  loss {total / len(moves):.4f}"
        if val is not None:
            line += f"  val accuracy {accuracy(model, val):.3f}"
        print(line, flush=True)
