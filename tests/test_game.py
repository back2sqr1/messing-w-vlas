from dodge.game import BLUE, RED, instruction


def test_instruction_names_the_bad_color():
    assert instruction(RED) == "dodge red catch blue"
    assert instruction(BLUE) == "dodge blue catch red"
    assert instruction(RED, 3) == "collect blue and avoid red"
