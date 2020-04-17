from memory import Memory
import gym
import numpy as np
import ray


@ray.remote
class A2CAgent:
    def __init__(self, num_steps, env):
        self.chief = False
        self.env = gym.make(env)
        self.num_steps = num_steps
        self.finished = False

        # environment parameters
        self.obs_space_size = self.env.observation_space.shape[0]
        self.action_space_size =  self.env.action_space.shape[0]
        self.action_space_bounds = [self.env.action_space.low[0], self.env.action_space.high[0]]

        # get initial state and initialize memory
        self.state = self.env.reset()
        self.memory = Memory(self.num_steps, self.obs_space_size, self.action_space_size)

    def observe(self, t):
        # reset memory for new network update
        if t == 0:
            self.memory.reset(self.num_steps, self.obs_space_size, self.action_space_size)

        # reset environment at the end of an episode
        if self.finished:
            self.state = self.env.reset()

        # render chief
        if self.chief:
            self.env.render()

        self.state = np.reshape(self.state, [1,self.obs_space_size])

        return self.state, self.finished

    def execute(self, action, t):

        if self.finished:
            self.finished = False

        # clip action value if necessary to be within action space
        action = np.clip(action, self.action_space_bounds[0], self.action_space_bounds[1])

        # perform action and store the resulting state and reward
        next_state, reward, done, _ = self.env.step(action)
        self.memory.store(self.state, action, reward, done, t)
        self.state = next_state
        self.finished = done


        if t == (self.num_steps - 1):
            return self.memory
        else:
            return None

    def set_chief(self):
        self.chief = True
