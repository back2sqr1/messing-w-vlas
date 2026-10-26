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


def live(policy, game, text, fps, swap, rng):
    """Window mode. Space flips which color is bad (rule and instruction together). Esc quits."""
    pygame.init()
    screen = pygame.display.set_mode(SIZE)
    pygame.display.set_caption("dodge VLA   space: flip colors   esc: quit")
    font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                game.bad = other(game.bad)
                text = told(game, swap, rng)
        draw(screen, font, game, text)
        pygame.display.flip()
        clock.tick(fps)
        if game.over:
            pygame.time.wait(1500)
            game = Game(game.bad, seed=int(rng.integers(2**31)))
            text = told(game, swap, rng)
        else:
            game.step(policy(game, text))


def main():
    p = argparse.ArgumentParser(description="Watch a policy play, or record one episode to MP4.")
    p.add_argument("--policy", choices=["vla", "expert", "random"], default="vla")
    p.add_argument("--model", default="out/vla.pt")
    p.add_argument("--bad", choices=["red", "blue"], help="color that crashes the robot (default: random)")
    p.add_argument("--swap", action="store_true", help="tell the policy the opposite color")
    p.add_argument("--seed", type=int, help="default: random")
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--record", metavar="MP4", help="write one episode to this file instead of opening a window")
    args = p.parse_args()
    seed = args.seed if args.seed is not None else int(np.random.default_rng().integers(2**31))
    rng = np.random.default_rng(seed)
    bad = {"red": RED, "blue": BLUE}[args.bad] if args.bad else int(rng.choice([RED, BLUE]))
    game = Game(bad, seed=int(rng.integers(2**31)))
    text = told(game, args.swap, rng)
    if args.policy == "vla":
        policy = vla_policy(load_vla(args.model))
    elif args.policy == "expert":
        policy = expert_policy
    else:
        policy = random_policy(seed)
    print(f"seed {seed}")
    if args.record:
        record(args.record, policy, game, text, args.fps)
        result = "crashed" if game.crashed else "survived"
        print(f"wrote {args.record}: {result} after {game.steps} steps, caught {game.caught}")
    else:
        live(policy, game, text, args.fps, args.swap, rng)


if __name__ == "__main__":
    main()
