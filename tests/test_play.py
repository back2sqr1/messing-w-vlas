import pygame

from dodge.game import RED, Game, instruction, render
from dodge.play import BANNER, SCALE, SIZE, draw


def test_draw_puts_the_upscaled_frame_under_the_banner():
    pygame.font.init()
    screen = pygame.Surface(SIZE)
    game = Game(RED, seed=0)
    draw(screen, pygame.font.Font(None, 24), game, instruction(RED))
    pixels = pygame.surfarray.array3d(screen).swapaxes(0, 1)  # back to (y, x, rgb)
    board = render(game.grid()).repeat(SCALE, axis=0).repeat(SCALE, axis=1)
    assert pixels.shape == (SIZE[1], SIZE[0], 3)
    assert (pixels[BANNER:] == board).all()
