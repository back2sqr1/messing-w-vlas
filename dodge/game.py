import numpy as np

EMPTY, RED, BLUE, ROBOT = 0, 1, 2, 3
LEFT, STAY, RIGHT = 0, 1, 2
COLS, ROWS = 8, 8  # the robot lives in the bottom row
CELL = 8  # pixels per cell, so a frame is 64x64
SPAWN_P = 0.5  # chance per step of a red block in the new top row; same for blue
MAX_STEPS = 200

NAMES = {RED: "red", BLUE: "blue"}
TEMPLATES = (
    "dodge {bad} catch {good}",
    "catch {good} dodge {bad}",
    "avoid {bad} and collect {good}",
    "collect {good} and avoid {bad}",
)


def other(color):
    return RED + BLUE - color


def instruction(bad, template=0):
    return TEMPLATES[template].format(bad=NAMES[bad], good=NAMES[other(bad)])
