import tensorflow as tf
from tensorflow_probability.python.distributions import Normal
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
        self.actor = None

    def get_action_distribution(self, state):
        mu, sigma = self.actor(state)
        return Normal(loc=mu[0], scale=sigma[0])
        
    def run(self, actor, critic, num_steps, test=False):
        self.num_steps = num_steps
        self.memory = Memory(self.num_steps, self.obs_space_size, self.action_space_size)
        self.actor = actor
        
        if self.finished:
            self.state = self.env.reset()
            self.finished = False

        # Agent has 500 trials at max, if it does not fail beforehand
        for t in range(self.num_steps):
            #if test:
                #self.env.render()
            self.state = np.reshape(self.state, [1,1,self.obs_space_size])
            # sample action from normal distribution
            action = self.get_action_distribution(self.state).sample()
            # Execute action and store action, state and reward
            action = tf.clip_by_value(action, self.action_space_bounds[0], self.action_space_bounds[1]) 
            
#            print(" before: ", action)
            
            # round action if action space is discrete
            if type(self.env.action_space).__name__ == 'Discrete':
                action = tf.math.round(action[0])
                action = tf.cast(action, tf.int32).numpy()
                
#            print(" after: ", action)
            
            next_state, reward, done, _ = self.env.step(action[0]) 
            self.memory.store(self.state, action, reward, done, t)
            self.state = next_state
            self.finished = done
            # Reset the environment if the agent terminates
            if done:
                self.state = self.env.reset()
            if test and done:
                break
        if not test:
            self.memory.compute_discounted_cum_return(critic)
        return self.memory