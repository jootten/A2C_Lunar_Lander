import numpy as np
import scipy.signal as signal

GAMMA = 0.99

class Memory:
    def __init__(self, num_steps, obs_space_size, action_space_size):
        self.obs_space_size = obs_space_size
        self.action_space_size = action_space_size
        self.states = np.empty(shape=(num_steps, obs_space_size))
        self.actions = np.empty(shape=(num_steps, action_space_size))
        self.rewards = np.zeros(shape=(num_steps, 1))
        self.estimated_return = np.empty(shape=(num_steps, 1))
        self.terminals = []
    
    def store(self, state, action, reward, done, t):
        self.states[t] = state
        self.actions[t] = action
        self.rewards[t] = reward
        self.terminals.append(done)

    def reset(self, num_steps, obs_space_size, action_space_size):
        self.__init__(num_steps, obs_space_size, action_space_size)

    def __add__(self, other):
        self.states = np.concatenate((self.states, other.states), axis=0)
        self.actions = np.concatenate((self.actions, other.actions), axis=0)
        self.rewards = np.concatenate((self.rewards, other.rewards), axis=0)
        self.estimated_return = np.concatenate((self.estimated_return, other.estimated_return), axis=0)
        self.terminals.append(other.terminals)
        return self

    def __radd__(self, other):
        if other == 0 or other == None:
            return self
        else:
            return self.__add__(other)

    def compute_discounted_cum_return(self, critic):
        idx = (len(self.rewards) - 1)
        if self.terminals[idx]:
            cumulative_return = 0  
        else:
            cumulative_return = critic(np.reshape(self.states[idx], [1,self.obs_space_size]))[0,0]
        
        for i in range(idx - 1, -1, -1):
            self.estimated_return[i][0] = self.rewards[i][0] + GAMMA * cumulative_return
            cumulative_return = 0 if self.terminals[i] else self.estimated_return[i][0]

            
        
