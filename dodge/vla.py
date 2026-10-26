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
