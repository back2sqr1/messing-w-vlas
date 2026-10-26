import torch
import torch.nn.functional as F
from torch import nn

from dodge.game import BLUE, CELL, COLS, RED, ROWS, TEMPLATES, instruction, render

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALL_INSTRUCTIONS = [instruction(bad, t) for bad in (RED, BLUE) for t in range(len(TEMPLATES))]
VOCAB = ["<pad>"] + sorted({word for text in ALL_INSTRUCTIONS for word in text.split()})
MAX_WORDS = max(len(text.split()) for text in ALL_INSTRUCTIONS)


def tokenize(text):
    ids = [VOCAB.index(word) for word in text.split()]  # unknown word -> ValueError
    return ids + [0] * (MAX_WORDS - len(ids))


class TinyVLA(nn.Module):
    """Frame + instruction -> one score per move (LEFT, STAY, RIGHT).

    Vision: one feature vector per game cell, refined by 3x3 convolutions.
    Language: the word vectors are laid side by side, so word order counts, and turned into a
    scale and a shift for every vision channel (FiLM, the same trick RT-1 uses). The text
    changes what the vision layers compute, such as whether a red cell is a thing to dodge.
    """

    def __init__(self, dim=64, layers=3):
        super().__init__()
        self.word = nn.Embedding(len(VOCAB), 32)
        self.text = nn.Sequential(nn.Linear(MAX_WORDS * 32, 128), nn.ReLU(), nn.Linear(128, 2 * dim * layers))
        self.patch = nn.Conv2d(3, dim, kernel_size=CELL, stride=CELL)  # one patch = one game cell
        self.convs = nn.ModuleList(nn.Conv2d(dim, dim, 3, padding=1) for _ in range(layers))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(dim * ROWS * COLS, 256), nn.ReLU(), nn.Linear(256, 3))

    def forward(self, frames, words):
        """frames: (B, 64, 64, 3) uint8, words: (B, MAX_WORDS) int64 -> (B, 3) move scores."""
        film = self.text(self.word(words).flatten(1)).chunk(2 * len(self.convs), dim=1)
        x = self.patch(frames.permute(0, 3, 1, 2).float() / 255)
        for i, conv in enumerate(self.convs):
            scale, shift = film[2 * i][:, :, None, None], film[2 * i + 1][:, :, None, None]
            x = x + conv(F.relu(x * (1 + scale) + shift))
        return self.head(F.relu(x))


@torch.no_grad()
def act(model, grid, text):
    device = next(model.parameters()).device
    frame = torch.as_tensor(render(grid)[None], device=device)
    words = torch.tensor([tokenize(text)], device=device)
    return int(model(frame, words).argmax())
