import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from multi_armed_bandits import Bandit


@pytest.mark.parametrize(("q_approximation", "n_actions", "expected"),
                         [
                             (np.array([1, 2, 4]), np.array([1, 1, 1]), 2),
                             (np.array([0]*6), np.array([0]*6), 5),
                             (np.array([1]*6), np.array([1]*6), 5),
                             (np.array([0, 0, 4, 2]), np.array([0, 0, 2, 2]), 2)
                         ])
def test_action_choose_action_return_correct_values(q_approximation, n_actions, expected):
    bandit = Bandit(n_bandits=len(q_approximation), seed=123)
    bandit.reset()
    bandit.q_approximation = q_approximation
    bandit.n_actions = n_actions
    result = bandit.act()
    assert_almost_equal(result, expected)


@pytest.fixture()
def sampled_rewards():
    return np.array([[1,2,3],[3,4,5],[6,7,8]])


@pytest.mark.parametrize(("actions", "expected_q_approximation", "expected_n_actions", "expected_rewards"),
                         [
                             ([1], np.array([0, 2, 0]), np.array([0, 1, 0]), 2),
                             ([1, 2], np.array([0, 2, 5]), np.array([0, 1, 1]), 5),
                             ([1, 1], np.array([0, 3, 0]), np.array([0, 2, 0]), 4),
                         ])
def test_action_step_return_correct_values(sampled_rewards, actions, expected_rewards, expected_n_actions,
                                           expected_q_approximation):
    bandit = Bandit(n_bandits=3, n_steps=3)
    bandit.reset()
    bandit.sampled_rewards = sampled_rewards
    rewards = None
    for act in actions:
        rewards = bandit.step(act)
    assert rewards == expected_rewards
    assert_almost_equal(bandit.n_actions, expected_n_actions)
    assert_almost_equal(bandit.q_approximation, expected_q_approximation)



@pytest.mark.parametrize(("epsilon", "n_simulations", "expected"),
                         [
                             (0,
                              1,
                              np.array([ 0.6246492, -1.3010917])),
                             (0,
                              5,
                              np.array([ 0.3415778, -0.1636272])),
                             (0.5,
                              5,
                              np.array([-0.1377239, -0.5231555])),
                         ]
                         )
def test_action_simulate_return_correct_rewards(epsilon, n_simulations, expected):
    bandit = Bandit(epsilon=epsilon, n_steps=2, n_bandits=3, n_simulations=n_simulations, seed=123)
    bandit.reset()
    result, _ = bandit.simulate()
    assert_almost_equal(expected, result)


@pytest.mark.parametrize(("epsilon", "n_simulations", "expected"),
                         [
                             (0,
                              1,
                              np.array([ 0, 0])),
                             (0,
                              5,
                              np.array([ 0.2, 0.4])),
                             (0.2,
                              100,
                              np.array([0.31, 0.46])),
                         ]
                         )
def test_action_simulate_return_correct_optimum_action(epsilon, n_simulations, expected):
    bandit = Bandit(epsilon=epsilon, n_steps=2, n_bandits=3, n_simulations=n_simulations, seed=123)
    bandit.reset()
    _, result = bandit.simulate()
    assert_almost_equal(result, expected)


@pytest.mark.parametrize(("epsilon", "n_simulations", "expected"),
                         [
                             (0,
                              1,
                              np.array([0.6246492, 2.0920771])),
                             (0,
                              5,
                              np.array([-0.1922424, -0.4709467])),
                             (0.5,
                              5,
                              np.array([-0.1377239, -0.5231555])),
                         ]
                         )
def test_action_simulate_with_ucb_return_correct_rewards(epsilon, n_simulations, expected):
    bandit = Bandit(epsilon=epsilon, n_steps=2, n_bandits=3, n_simulations=n_simulations, seed=123, ucb=0.2)
    bandit.reset()
    result, _ = bandit.simulate()
    assert_almost_equal(expected, result)
