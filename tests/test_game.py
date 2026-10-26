import numpy as np

from dodge.game import (BLUE, CELL, COLS, EMPTY, LEFT, MAX_STEPS, RED, RIGHT, ROBOT, ROWS, SPRITES, STAY, Game,
                        instruction, render)


def test_instruction_names_the_bad_color():
    assert instruction(RED) == "dodge red catch blue"
    assert instruction(BLUE) == "dodge blue catch red"
    assert instruction(RED, 3) == "collect blue and avoid red"


def empty_game(bad=RED, robot=3):
    game = Game(bad, seed=0)
    game.blocks[:] = EMPTY
    game.robot = robot
    return game


def test_robot_moves_one_column_and_stops_at_walls():
    game = empty_game(robot=0)
    game.step(LEFT)
    assert game.robot == 0
    game.step(RIGHT)
    assert game.robot == 1
    game.step(STAY)
    assert game.robot == 1


def test_blocks_fall_one_row_per_step():
    game = empty_game(robot=0)
    game.blocks[2, 5] = BLUE
    game.step(STAY)
    assert game.blocks[3, 5] == BLUE and game.blocks[2, 5] == EMPTY


def test_bad_block_landing_on_robot_ends_the_episode():
    game = empty_game(bad=RED, robot=3)
    game.blocks[-1, 3] = RED
    game.step(STAY)
    assert game.crashed and game.over


def test_moving_out_from_under_a_bad_block_avoids_it():
    game = empty_game(bad=RED, robot=3)
    game.blocks[-1, 3] = RED
    game.step(LEFT)
    assert game.robot == 2 and not game.crashed


def test_good_block_landing_on_robot_is_caught():
    game = empty_game(bad=RED, robot=3)
    game.blocks[-1, 3] = BLUE
    game.step(STAY)
    assert game.caught == 1 and not game.over


def test_episode_ends_after_max_steps():
    game = empty_game()
    game.steps = MAX_STEPS - 1
    game.step(STAY)
    assert game.over and not game.crashed


def test_new_rows_hold_at_most_one_block_of_each_color():
    game = Game(RED, seed=1)
    reds = 0
    for _ in range(2000):
        game.fall()
        assert (game.blocks[0] == RED).sum() <= 1 and (game.blocks[0] == BLUE).sum() <= 1
        reds += (game.blocks[0] == RED).sum()
    assert 0.4 < reds / 2000 < 0.6  # SPAWN_P = 0.5


def test_render_draws_each_cell_as_its_sprite():
    grid = np.zeros((ROWS, COLS), np.int8)
    grid[0, 0], grid[2, 5], grid[-1, 4] = RED, BLUE, ROBOT
    frame = render(grid)
    assert frame.shape == (ROWS * CELL, COLS * CELL, 3) and frame.dtype == np.uint8
    assert (frame[:CELL, :CELL] == SPRITES[RED]).all()
    assert (frame[2 * CELL:3 * CELL, 5 * CELL:6 * CELL] == SPRITES[BLUE]).all()
    assert (frame[-CELL:, 4 * CELL:5 * CELL] == SPRITES[ROBOT]).all()
    assert (frame[CELL:2 * CELL, :CELL] == SPRITES[EMPTY]).all()


def test_render_handles_a_batch_of_grids():
    grids = np.stack([Game(RED, seed=s).grid() for s in range(5)])
    frames = render(grids)
    assert frames.shape == (5, ROWS * CELL, COLS * CELL, 3)
    assert all((frames[i] == render(grids[i])).all() for i in range(5))
