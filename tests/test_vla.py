import numpy as np
import pytest
import torch

from dodge.game import BLUE, LEFT, RED, RIGHT, Game, instruction, render
from dodge.vla import MAX_WORDS, VOCAB, TinyVLA, act, tokenize


def test_tokenize_pads_to_max_words():
    ids = tokenize("dodge red catch blue")
    assert len(ids) == MAX_WORDS
    assert [VOCAB[i] for i in ids] == ["dodge", "red", "catch", "blue"] + ["<pad>"] * (MAX_WORDS - 4)


def test_tokenize_rejects_unknown_words():
    with pytest.raises(ValueError):
        tokenize("dodge green")


def test_model_returns_three_move_scores_per_frame():
    frames = torch.as_tensor(render(np.stack([Game(RED, seed=s).grid() for s in range(2)])))
    words = torch.tensor([tokenize(instruction(RED)), tokenize(instruction(BLUE))])
    assert TinyVLA()(frames, words).shape == (2, 3)


def test_model_can_learn_to_follow_word_order():
    # Same frame, same bag of words, different order, different correct move:
    # this only fits if the text, including word order, reaches the output.
    torch.manual_seed(0)
    model = TinyVLA()
    frames = torch.as_tensor(render(Game(RED, seed=0).grid())).expand(2, -1, -1, -1)
    words = torch.tensor([tokenize("dodge red catch blue"), tokenize("dodge blue catch red")])
    target = torch.tensor([LEFT, RIGHT])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(30):
        loss = torch.nn.functional.cross_entropy(model(frames, words), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert model(frames, words).argmax(1).tolist() == [LEFT, RIGHT]


def test_act_returns_a_move():
    assert act(TinyVLA(), Game(RED, seed=0).grid(), instruction(RED)) in (0, 1, 2)
