import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from gridworld import Gridworld


@pytest.fixture()
def gridworld():
    height = 5
    width = 5
    state_A = (0, 1)
    state_B = (0, 3)
    from_A = (4, 1)
    from_B = (2, 3)
    reward_leaving_state_A = 10
    reward_leaving_state_B = 5
    gamma = 0.9
    return Gridworld(height, width,
                     state_A, state_B,
                     from_A, from_B,
                     reward_leaving_state_A, reward_leaving_state_B,
                     gamma)


def test_get_state_value_correctly(gridworld):
    expected = np.array([[3.3, 8.8, 4.4, 5.3, 1.5],
                         [1.5, 3.0, 2.3, 1.9, 0.5],
                         [0.1, 0.7, 0.7, 0.4, -0.4],
                         [-1.0, -0.4, -0.4, -0.6, -1.2],
                         [-1.9, -1.3, -1.2, -1.4, -2.0]
                         ])
    assert_almost_equal(np.round(gridworld.state_values, decimals=1), expected)


def test_get_optimum_state_value_correctly(gridworld):
    gridworld.build_optimum_state_value()
    expected = np.array([[22.0, 24.4, 22.0, 19.4, 17.5],
                         [19.8, 22.0, 19.8, 17.8, 16.0],
                         [17.8, 19.8, 17.8, 16.0, 14.4],
                         [16.0, 17.8, 16.0, 14.4, 13.0],
                         [14.4, 16.0, 14.4, 13.0, 11.7]
                         ])
    assert_almost_equal(np.round(gridworld.state_values, decimals=1), expected)
