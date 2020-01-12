import numpy as np


class Gridworld:
    def __init__(self, height: int, width: int,
                 state_A: tuple, state_B: tuple,
                 from_A: tuple, from_B: tuple,
                 reward_leaving_A: float, reward_leaving_B: float, gamma: float):
        self.state_A = state_A
        self.from_A = from_A
        self.state_B = state_B
        self.from_B = from_B
        self.gamma = gamma
        assert self.state_A != self.state_B
        self.reward_leaving_A = reward_leaving_A
        self.reward_leaving_B = reward_leaving_B
        self.reward_leaving_board = -1.
        self._height = height
        self._width = width
        self.state_values = np.zeros((height, width))
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.build_state_value()

    def step(self, action, state):
        if state == self.state_A:
            return self.from_A, self.reward_leaving_A

        if state == self.state_B:
            return self.from_B, self.reward_leaving_B

        next_state = (state[0]+action[0], state[1]+action[1])

        if 0 <= next_state[0] < self._height and 0 <= next_state[1] < self._width:
            return next_state, 0.

        return state, -1.

    def build_state_value(self):

        last_state_values = np.random.randn(self._height, self._width)
        state_values = np.zeros((self._height, self._width))
        difference = 0.1
        while difference > 0.0000001:
            for i in range(self._height):
                for j in range(self._width):
                    state = (i, j)
                    return_value = 0.0
                    for action in self.actions:
                        next_state, reward = self.step(action, state)
                        return_value += self.get_action_probabilities(action, state) * (
                                reward + self.gamma * last_state_values[next_state[0], next_state[1]]
                        )
                    state_values[i, j] = return_value
            difference = np.abs(np.sum(state_values - last_state_values))
            last_state_values = state_values.copy()

        self.state_values = last_state_values

    def get_action_probabilities(self, action: tuple, state: tuple):
        return 1./len(self.actions)

    def build_optimum_state_value(self):
        last_state_values = np.ones((self._height, self._width))
        state_values = np.zeros((self._height, self._width))
        difference = 0.1
        while difference > 1e-5:
            for i in range(self._height):
                for j in range(self._width):
                    state = (i, j)
                    return_values = []
                    for action in self.actions:
                        next_state, reward = self.step(action, state)
                        return_values.append(reward + self.gamma * last_state_values[next_state[0], next_state[1]])
                    state_values[i, j] = np.max(return_values)
            difference = np.abs(np.sum(state_values - last_state_values))
            last_state_values = state_values.copy()

        self.state_values = last_state_values
