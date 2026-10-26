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


def record(path, policy, game, text, fps):
    """Play one episode into an MP4 through the system ffmpeg. No window opens."""
    pygame.font.init()
    font = pygame.font.Font(None, 24)
    screen = pygame.Surface(SIZE)
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{SIZE[0]}x{SIZE[1]}", "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE,
    )
    while True:
        draw(screen, font, game, text)
        frame = pygame.image.tobytes(screen, "RGB")
        ffmpeg.stdin.write(frame)
        if game.over:
            break
        game.step(policy(game, text))
    ffmpeg.stdin.write(frame * fps)  # hold the last frame for one second
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise RuntimeError("ffmpeg failed")
