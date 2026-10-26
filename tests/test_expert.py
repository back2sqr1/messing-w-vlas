import itertools

import numpy as np

from dodge.expert import ORDER, expert_move, plan
from dodge.game import BLUE, COLS, LEFT, RED, RIGHT, ROWS, STAY, Game, other


def brute_force(blocks, robot, bad):
    """Try every move sequence over the visible rows; best score for each first move in ORDER."""
    rows = blocks.tolist()
    best = {m: -np.inf for m in ORDER}
    for moves in itertools.product(ORDER, repeat=len(rows)):
        x, score = robot, 0.0
        for k, m in enumerate(moves):
            x = min(max(x + m - 1, 0), COLS - 1)
            cell = rows[-1 - k][x]  # the bottom row lands after the first move
            score += -np.inf if cell == bad else float(cell == other(bad))
        best[moves[0]] = max(best[moves[0]], score)
    return [best[m] for m in ORDER]


def blank():
    return np.zeros((ROWS - 1, COLS), np.int8)


def test_expert_steps_out_from_under_a_bad_block():
    blocks = blank()
    blocks[-1, 3] = RED
    assert expert_move(blocks, 3, bad=RED) == LEFT  # LEFT and RIGHT tie; LEFT comes first in ORDER


def test_expert_moves_under_a_good_block():
    blocks = blank()
    blocks[-1, 4] = BLUE
    assert expert_move(blocks, 3, bad=RED) == RIGHT


def test_expert_stays_on_an_empty_board():
    assert expert_move(blank(), 3, bad=RED) == STAY


def test_expert_reads_the_task():
    blocks = blank()
    blocks[-1, 3], blocks[-1, 4] = RED, BLUE  # both land next step
    assert expert_move(blocks, 3, bad=RED) == RIGHT  # dodge red, catch blue
    assert expert_move(blocks, 3, bad=BLUE) == STAY  # dodge blue, catch red


def test_plan_matches_brute_force_on_random_boards():
    rng = np.random.default_rng(0)
    for _ in range(100):
        game = Game(bad=int(rng.choice([RED, BLUE])), seed=int(rng.integers(2**31)))
        assert plan(game.blocks, game.robot, game.bad) == brute_force(game.blocks, game.robot, game.bad)
