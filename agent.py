import os
#import tensorflow as tf
#from tensorflow_probability.python.distributions import Normal
import gym
import numpy as np
import ray
from memory import Memory

# Now as parameter for Coordinator init
#ENV = 'LunarLanderContinuous-v2'
#ENV = 'CartPole-v1'

@ray.remote
class A2CAgent:
    def __init__(self, num_steps, env):
        self.env = env
        self.obs_space_size = self.env.observation_space.shape[0]
        self.action_space_size = 1 if self.env.action_space.shape == () else self.env.action_space.shape[0]
        self.action_space_bounds = [0, self.env.action_space.n-1] if self.env.action_space.shape == () else [self.env.action_space.low[0], self.env.action_space.high[0]]
        self.num_steps = num_steps
        self.finished = True
        self.state = None
        #self.actor = None
        #self.previous_recurrent_state = [None, None]
        self.memory = Memory(self.num_steps, self.obs_space_size, self.action_space_size)

    #def get_action_distribution(self, state):
    #    mu, sigma, self.previous_recurrent_state = self.actor(state, initial_state=self.previous_recurrent_state)
    #    return Normal(loc=mu, scale=sigma)
        
    def observe(self, s):#, actor, critic, test=False):
        if s == 0:
            self.memory.reset(self.num_steps, self.obs_space_size, self.action_space_size)
        #self.actor = actor
        
        # Reset the environment if the agent terminates
        if self.finished:
            self.state = self.env.reset()
            self.finished = False
            #self.previous_recurrent_state = [None, None]

        # Agent has 500 trials at max, if it does not fail beforehand
        #for t in range(self.num_steps):
        #if test:
            #self.env.render()
        self.state = np.reshape(self.state, [1,self.obs_space_size])

        return self.state
        
    def execute(self, action, s):
        # sample action from normal distribution
        #action = self.get_action_distribution(self.state).sample()[0]
        # Execute action and store action, state and reward
        action = np.clip(action, self.action_space_bounds[0], self.action_space_bounds[1]) 
        
#           print(" before: ", action)
        
        # round action if action space is discrete
        #if type(self.env.action_space).__name__ == 'Discrete':
        #    action = np.math.round(action)
        #    action = tf.cast(action, tf.int32).numpy()
            
#           print(" after: ", action)
        
        next_state, reward, done, _ = self.env.step(action) 
        self.memory.store(self.state, action, reward, done, s)
        self.state = next_state
        self.finished = done
        
        
        if s == (self.num_steps - 1):
            return self.memory
        else:
            return None

    
