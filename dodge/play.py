import argparse
import os
import subprocess

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")  # otherwise pygame prints a banner on import
import pygame

from dodge.evaluate import expert_policy, load_vla, random_policy, told, vla_policy
from dodge.game import BLUE, CELL, COLS, NAMES, RED, ROWS, Game, other, render

SCALE = 8  # the 64x64 frame is shown at 512x512
BANNER = 48  # text strip above the board; 512x560 keeps both sides even, which H.264 needs
SIZE = (COLS * CELL * SCALE, ROWS * CELL * SCALE + BANNER)


def draw(screen, font, game, text):
    board = render(game.grid()).repeat(SCALE, axis=0).repeat(SCALE, axis=1)
    screen.fill((0, 0, 0))
    screen.blit(pygame.surfarray.make_surface(board.swapaxes(0, 1)), (0, BANNER))  # pygame indexes (x, y)
    status = f"real rule: dodge {NAMES[game.bad]}   caught {game.caught}   step {game.steps}"
    if game.crashed:
        status += "   CRASH"
    screen.blit(font.render(f'told: "{text}"', True, (255, 255, 255)), (10, 6))
    screen.blit(font.render(status, True, (170, 170, 170)), (10, 26))
