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


def main():
    p = argparse.ArgumentParser(description="Score the random, expert and VLA policies on the same games.")
    p.add_argument("--model", default="out/vla.pt")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, help="default: random")
    args = p.parse_args()
    seed = args.seed if args.seed is not None else int(np.random.default_rng().integers(2**31))
    vla = vla_policy(load_vla(args.model))
    rows = [
        ("random", random_policy(seed), False),
        ("expert", expert_policy, False),
        ("vla", vla, False),
        ("vla, told the opposite color", vla, True),
    ]
    print(f"seed {seed}, {args.episodes} episodes of up to {MAX_STEPS} steps")
    print(f"{'policy':30} {'survived':>8} {'caught/episode':>15}")
    for name, policy, swap in rows:
        r = evaluate(policy, args.episodes, seed, swap)
        print(f"{name:30} {r['survived']:8.2f} {r['caught']:15.1f}", flush=True)


if __name__ == "__main__":
    main()
