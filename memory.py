import numpy as np

class Memory:
    def __init__(self, num_steps):
        self.states = np.empty(shape=(num_steps, 8))
        self.actions = np.empty(shape=(num_steps, 2))
        self.rewards = np.zeros(shape=(num_steps, 1))
        self.terminals = []
    
    def store(self, state, action, reward, done, t):
        self.states[t] = state
        self.actions[t] = action
        self.rewards[t] = reward
        self.terminals.append(done)

    def __add__(self, other):
        self.states = np.concatenate((self.states, other.states), axis=0)
        self.actions = np.concatenate((self.actions, other.actions), axis=0)
        self.rewards = np.concatenate((self.rewards, other.rewards), axis=0)
        self.terminals.append(other.terminals)
        return self

    def __radd__(self, other):
        if other == 0 or other == None:
            return self
        else:
            return self.__add__(other)
