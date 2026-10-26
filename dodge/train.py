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


def main():
    p = argparse.ArgumentParser(description="Play expert games, then train the VLA to copy the expert's moves.")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seed", type=int, help="default: random")
    p.add_argument("--out", default="out/vla.pt")
    args = p.parse_args()
    seed = args.seed if args.seed is not None else int(np.random.default_rng().integers(2**31))
    torch.manual_seed(seed)
    print(f"seed {seed}  device {DEVICE}", flush=True)
    start = time.time()
    data = collect(args.episodes, seed)
    val = collect(max(args.episodes // 10, 1), seed + 1)
    print(f"{len(data[2])} training frames, {len(val[2])} validation frames ({time.time() - start:.0f}s)", flush=True)
    model = TinyVLA().to(DEVICE)
    train(model, data, args.epochs, seed, val=val)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"saved {args.out}  ({time.time() - start:.0f}s total)")


if __name__ == "__main__":
    main()
