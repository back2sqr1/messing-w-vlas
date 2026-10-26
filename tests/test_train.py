import numpy as np

from dodge.expert import expert_move
from dodge.game import TEMPLATES, instruction
from dodge.train import collect
from dodge.vla import VOCAB


def test_collect_labels_every_frame_once_per_task():
    grids, words, moves, bads = collect(episodes=2, seed=0)
    assert len(grids) == len(words) == len(moves) == len(bads) > 0
    assert (grids[0::2] == grids[1::2]).all()  # each frame appears twice ...
    assert (bads[0::2] != bads[1::2]).all()  # ... once with each bad color
    for grid, bad, move in zip(grids[:100], bads[:100], moves[:100]):
        robot = int(np.flatnonzero(grid[-1])[0])
        assert move == expert_move(grid[:-1], robot, bad)


def test_each_label_comes_with_an_instruction_for_its_task():
    _, words, _, bads = collect(episodes=1, seed=0)
    for ids, bad in zip(words, bads):
        text = " ".join(VOCAB[i] for i in ids if i)
        assert text in [instruction(bad, t) for t in range(len(TEMPLATES))]
