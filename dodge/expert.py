import numpy as np

from dodge.game import LEFT, RIGHT, STAY, other

ORDER = (STAY, LEFT, RIGHT)  # ties go to the earlier move, so every frame has one fixed label


def plan(blocks, robot, bad):
    """Best score for each first move in ORDER, counting every visible block.

    Score: +1 per good block caught, -inf if a bad block lands on the robot.
    Blocks that have not appeared yet count as nothing.
    """
    cols = blocks.shape[1]
    value = np.zeros(cols)  # value[x]: best score from here on with the robot at column x
    for row in blocks:  # the top row lands last, so it is added first
        reach = np.maximum(np.maximum(np.r_[value[0], value[:-1]], value), np.r_[value[1:], value[-1]])
        value = np.where(row == bad, -np.inf, (row == other(bad)).astype(float)) + reach
    return [value[min(max(robot + m - 1, 0), cols - 1)] for m in ORDER]


def expert_move(blocks, robot, bad):
    return ORDER[int(np.argmax(plan(blocks, robot, bad)))]
