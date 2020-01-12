import numpy as np
from tqdm import trange
from math import sqrt, log

class Bandit:
    def __init__(self, n_bandits=10, n_steps=1000, n_simulations=2000,epsilon=0, initial=None, step_size=None, seed=None,
                 ucb=None):
        self.epsilon = epsilon
        self.seed = seed
        self.n_bandits = n_bandits
        self.step_size = step_size
        self.n_steps = n_steps
        self.n_simulations = n_simulations
        self.ucb = ucb
        if initial is None:
            self.initial = 0
        else:
            self.initial = initial

        self.q_distributions = None
        self.q_approximation = None
        self.n_actions = None
        self.sampled_rewards = None
        self.best_action = None
        self.stepwise_rewards = None
        self.time = 0

        if seed is not None:
            np.random.seed(seed)

    def reset(self):
        self.q_distributions = np.random.randn(self.n_bandits)
        self.q_approximation = np.zeros(self.n_bandits) + self.initial
        self.n_actions = np.zeros(self.n_bandits)
        self.sampled_rewards = np.random.randn(self.n_steps, self.n_bandits)+self.q_distributions
        self.best_action = np.argmax(self.q_distributions)
        self.stepwise_rewards = np.zeros(self.n_steps)
        self.time = 0

    def act(self) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(self.n_bandits))
        if self.ucb is not None:
            ucb = self.q_approximation+self.ucb*sqrt(log(self.time+1))/np.sqrt(self.n_actions+1e-5)
            best = np.max(ucb)
            return np.random.choice(np.where(ucb == best)[0])
        else:
            best = np.max(self.q_approximation)
            return np.random.choice(np.where(self.q_approximation == best)[0])

    def step(self, action: int) -> np.array:
        self.n_actions[action] += 1
        reward = self.sampled_rewards[self.time, action]
        self.time += 1
        if self.step_size is None:
            self.get_average_weighted_q_approximation(reward, action)
        else:
            self.get_step_size_q_approximation(reward, action)

        return reward

    def get_average_weighted_q_approximation(self, reward, action):
        self.q_approximation[action] += (reward-self.q_approximation[action])/self.n_actions[action]

    def get_step_size_q_approximation(self, reward, action):
        self.q_approximation[action] += self.step_size*(reward-self.q_approximation[action])

    def simulate(self):
        rewards = np.zeros((self.n_simulations, self.n_steps))
        optimum_action = np.zeros((self.n_simulations, self.n_steps))
        for i in trange(self.n_simulations):
            self.reset()
            for time in range(self.n_steps):
                action = self.act()
                if action == self.best_action:
                    optimum_action[i, time] = 1
                new_reward = self.step(action)
                rewards[i, time] = new_reward

        return np.mean(rewards, axis=0), np.mean(optimum_action, axis=0)
