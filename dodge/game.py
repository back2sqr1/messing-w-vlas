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


class Game:
    def __init__(self, bad, seed=None):
        self.bad = bad
        self.rng = np.random.default_rng(seed)
        self.blocks = np.zeros((ROWS - 1, COLS), np.int8)  # the last row lands on the robot row next step
        for _ in range(ROWS - 1):  # start with a full board, not an empty one
            self.fall()
        self.robot = int(self.rng.integers(COLS))
        self.steps = 0
        self.caught = 0
        self.crashed = False

    @property
    def over(self):
        return self.crashed or self.steps >= MAX_STEPS

    def fall(self):
        # At most one block of each color per row. The robot can always reach at least two cells
        # next step and at most one of them holds a bad block, so a safe move always exists.
        row = np.zeros(COLS, np.int8)
        red_col, blue_col = self.rng.permutation(COLS)[:2]
        if self.rng.random() < SPAWN_P:
            row[red_col] = RED
        if self.rng.random() < SPAWN_P:
            row[blue_col] = BLUE
        self.blocks = np.vstack([row, self.blocks[:-1]])

    def grid(self):
        g = np.zeros((ROWS, COLS), np.int8)
        g[:-1] = self.blocks
        g[-1, self.robot] = ROBOT
        return g

    def step(self, move):
        """Move the robot, then every block falls one row."""
        self.robot = min(max(self.robot + move - 1, 0), COLS - 1)
        landed = self.blocks[-1, self.robot]
        self.caught += int(landed == other(self.bad))
        self.crashed = bool(landed == self.bad)
        self.fall()
        self.steps += 1


PALETTE = {
    ".": (20, 20, 30),  # background
    "r": (230, 60, 60),
    "b": (60, 120, 240),
    "w": (225, 225, 225),  # robot body
    "y": (255, 200, 0),  # robot light
    "k": (70, 70, 80),  # robot eyes and wheels
}
BLOCK = ["........"] + [".######."] * 6 + ["........"]
ART = [
    ["........"] * 8,  # EMPTY
    [line.replace("#", "r") for line in BLOCK],  # RED
    [line.replace("#", "b") for line in BLOCK],  # BLUE
    [  # ROBOT
        "...yy...",
        "..wwww..",
        ".wkwwkw.",
        ".wwwwww.",
        ".wwwwww.",
        "..w..w..",
        ".kk..kk.",
        ".kk..kk.",
    ],
]
SPRITES = np.array([[[PALETTE[c] for c in line] for line in art] for art in ART], np.uint8)


def render(grids):
    """Cell grids (..., ROWS, COLS) -> RGB frames (..., ROWS*CELL, COLS*CELL, 3), uint8."""
    tiles = np.swapaxes(SPRITES[grids], -4, -3)  # (..., ROWS, CELL, COLS, CELL, 3)
    return tiles.reshape(*grids.shape[:-2], ROWS * CELL, COLS * CELL, 3)
