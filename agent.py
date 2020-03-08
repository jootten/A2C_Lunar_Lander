import tensorflow as tf
from tensorflow_probability.python.distributions import Normal
from actor import Actor
from critic import Critic
import gym
import numpy as np
import ray
from memory import Memory

@ray.remote
class A2CAgent:
    def __init__(self, chief=False):
        self.env = gym.make('LunarLanderContinuous-v2')
        
        self.gamma = 0.95 # discount factor
        self.num_steps = 5
        self.memory = Memory(self.num_steps)
        self.step = 0
        self.chief = chief
        self.actor = Actor()
        self.critic = Critic()
        self.finished = True
        self.state = None

    def get_action_distribution(self, state):
        mu, sigma = self.actor(state)
        return Normal(loc=mu, scale=sigma)
        
    def run(self, actor, critic, num_steps, test=False):
        self.actor = actor
        self.critic = critic
        self.num_steps = num_steps
        self.memory = Memory(num_steps)
        if self.finished or test:
            self.state = self.env.reset()
            self.finished = False
        
        # Agent has 500 trials at max, if it does not fail beforehand
        for t in range(self.num_steps):
            if test:
                self.env.render()
            self.state = np.reshape(self.state, [1,8])
            # sample action from normal distribution
            action = self.get_action_distribution(self.state).sample()
            # Execute action and store action, state and reward 
            next_state, reward, done, _ = self.env.step(action[0])  # TODO Implement new storage system (array for each variable
            self.memory.store(self.state, action, reward, done, t)             # since the gradients are just accumulated and not applied after each iter)
            self.state = next_state
            self.finished = done
            # Interrupt the trial if the agent fails
            if done:
                self.state = self.env.reset()
            # Compute gradients in training
            if not test and t == self.num_steps - 1:
                return self.memory
            self.step += 1