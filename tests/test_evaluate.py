import torch

from dodge.evaluate import evaluate, expert_policy, load_vla, random_policy
from dodge.game import RED, TEMPLATES, Game, instruction, other, render
from dodge.vla import TinyVLA, tokenize


def test_expert_survives_every_episode():
    assert evaluate(expert_policy, episodes=20, seed=0)["survived"] == 1.0


def test_random_moves_almost_always_crash():
    assert evaluate(random_policy(seed=0), episodes=20, seed=0)["survived"] < 0.2


def test_swap_tells_the_policy_the_opposite_color():
    seen = []

    def spy(game, text):
        seen.append((game.bad, text))
        return expert_policy(game, text)

    evaluate(spy, episodes=2, seed=0, swap=True)
    assert seen and all(text in [instruction(other(bad), t) for t in range(len(TEMPLATES))] for bad, text in seen)


def test_load_vla_restores_saved_weights(tmp_path):
    torch.manual_seed(0)
    model = TinyVLA()
    torch.save(model.state_dict(), tmp_path / "vla.pt")
    loaded = load_vla(tmp_path / "vla.pt").cpu()
    frame = torch.as_tensor(render(Game(RED, seed=0).grid())[None])
    words = torch.tensor([tokenize(instruction(RED))])
    assert torch.equal(model(frame, words), loaded(frame, words))
