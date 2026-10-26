import subprocess

import pygame

from dodge.evaluate import expert_policy
from dodge.game import MAX_STEPS, RED, Game, instruction, render
from dodge.play import BANNER, SCALE, SIZE, draw, record


def test_draw_puts_the_upscaled_frame_under_the_banner():
    pygame.font.init()
    screen = pygame.Surface(SIZE)
    game = Game(RED, seed=0)
    draw(screen, pygame.font.Font(None, 24), game, instruction(RED))
    pixels = pygame.surfarray.array3d(screen).swapaxes(0, 1)  # back to (y, x, rgb)
    board = render(game.grid()).repeat(SCALE, axis=0).repeat(SCALE, axis=1)
    assert pixels.shape == (SIZE[1], SIZE[0], 3)
    assert (pixels[BANNER:] == board).all()


def test_record_writes_every_step_plus_a_one_second_hold(tmp_path):
    game = Game(RED, seed=0)
    game.steps = MAX_STEPS - 3  # three moves left
    path = tmp_path / "clip.mp4"
    record(path, expert_policy, game, instruction(RED), fps=6)
    frames = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert frames == "10"  # first frame + 3 moves + 6 held frames
