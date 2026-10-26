# dodge-vla

A small vision-language-action model (VLA) that plays a falling-blocks game. Each step it gets
the 64x64 game frame and an instruction such as "dodge red catch blue", and picks a move: left,
stay or right. A bad-colored block landing on the robot ends the game; a good-colored one counts
as a catch. The model learns by copying a scripted expert that plans over every visible block.

## Setup

    uv sync
    uv run pytest -q

## Use

    uv run python -m dodge.train        # about 1 min on an RTX 3060 laptop GPU, writes out/vla.pt
    uv run python -m dodge.evaluate     # random vs expert vs VLA on the same 100 games
    uv run python -m dodge.play         # live window: space flips the colors, esc quits
    uv run python -m dodge.play --record out/vla.mp4
    uv run python -m dodge.play --swap --fps 3 --record out/vla_swapped.mp4   # told the wrong color

Every command takes `--seed`. Without it, a random seed is picked and printed.

## Results

`uv run python -m dodge.evaluate --seed 0`, model from `uv run python -m dodge.train --seed 0`:

    policy                         survived  caught/episode
    random                             0.00             0.8
    expert                             1.00            61.0
    vla                                0.98            59.8
    vla, told the opposite color       0.00             0.0

"survived" is the share of 100 games that reach 200 steps with no bad block landing on the robot.

## Model

    frame 64x64 -> 8x8 conv, stride 8 (one vector per cell) -> 3 x [FiLM, 3x3 conv] -> flatten -> MLP -> LEFT / STAY / RIGHT
    instruction -> word vectors side by side -> MLP -> the scale and shift used by each FiLM step

FiLM: the text sets a scale and a shift for every vision channel, the same way RT-1 conditions
its image encoder on language. Every training frame is stored with both instructions and the
expert's move for each, so the model has to read the text.

## Files

    dodge/game.py      grid game, instruction wording, drawing
    dodge/expert.py    scripted player: best move over the visible rows
    dodge/vla.py       TinyVLA: frame + instruction -> move
    dodge/train.py     record expert games, train by copying the expert
    dodge/evaluate.py  survival and catch numbers
    dodge/play.py      pygame window and MP4 recording
