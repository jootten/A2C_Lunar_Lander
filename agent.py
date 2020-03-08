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
        self.mse = tf.keras.losses.MSE
        self.step = 0
        self.chief = chief
        self.actor = Actor()
        self.critic = Critic()
        self.estimated_return = np.zeros(shape=(self.num_steps, 1))
        self.finished = True
        self.state = None

    def get_action_distribution(self, state):
        mu, sigma = self.actor(state)
        return Normal(loc=mu, scale=sigma)


    def _actor_loss(self):
        # Compute state value
        state_v = self.critic(self.memory.states)
        # Get advantage estimate
        advantages = self.estimated_return - state_v
        advantages = tf.cast(advantages, tf.float32)
        # Get log probability of the taken action
        logprob = self.get_action_distribution(self.memory.states).log_prob(self.memory.actions)
        # Advantage as baseline
        return -logprob * advantages

    def _compute_gradients(self, type):
        with tf.GradientTape() as tape:
            if type == 'actor':
                # Compute the actor loss
                loss = self._actor_loss()
            else:
                # Compute the statue value
                state_v = self.critic(self.memory.states)
                # Compute the critic loss
                loss = self.mse(self.estimated_return, state_v)
            # Compute the gradients
            return tape.gradient(loss, self.actor.trainable_variables if type == 'actor' else self.critic.trainable_variables)
        
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
                return self._get_mean_gradients()
            self.step += 1
    
    def _get_mean_gradients(self):
        # Lists to store gradients 
        policy_gradient_list = []
        critic_gradient_list = []
        # Iterate over taken actions and observed states and rewards
        # Compute estimated return
        self.estimated_return = self.gamma * self.estimated_return + self.memory.rewards
        # Compute gradients for the actor (policy gradient), Maximize the estimated return
        policy_gradient_list.append(self._compute_gradients('actor'))
        # Compute gradients for the critic, minimize MSE for the state value function
        critic_gradient_list.append(self._compute_gradients('critic'))
        # Compute mean gradients
        policy_gradients = np.mean(policy_gradient_list, axis=0)
        critic_gradients = np.mean(critic_gradient_list, axis=0)
        return policy_gradients, critic_gradients