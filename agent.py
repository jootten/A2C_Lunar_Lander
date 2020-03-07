import tensorflow as tf
from tensorflow_probability.python.distributions import Normal
from actor import Actor
from critic import Critic
import gym
import numpy as np

class A2CAgent:
    def __init__(self, chief=False):
        self.env = gym.make('LunarLanderContinuous-v2')
        
        self.gamma = 0.95 # discount factor
        self.observations = []
        self.mse = tf.keras.losses.MSE
        self.step = 0
        self.chief = chief
        self.actor = Actor()
        self.critic = Critic()
        self.estimated_return = 0

    def get_action_distribution(self, state):
        mu, sigma = self.actor(state)
        return Normal(loc=mu, scale=sigma)


    def _actor_loss(self, state, action):
        # Compute state value
        state_v = self.critic(state)
        # Get advantage estimate
        advantages = self.estimated_return - state_v
        advantages = tf.cast([[advantages]], tf.float32)
        # Get log probability of the taken action
        logprob = self.get_action_distribution(state).log_prob(action)
        # Advantage as baseline
        return logprob * advantages

    def _compute_gradients(self, type, state, action):
        with tf.GradientTape() as tape:
            if type == 'actor':
                # Compute the actor loss
                loss = self._actor_loss(state, action)
            else:
                # Compute the statue value
                state_v = self.critic(state)
                # Compute the critic loss
                loss = self.mse(self.estimated_return, state_v)
            # Compute the gradients
            return tape.gradient(loss, self.actor.trainable_variables if type == 'actor' else self.critic.trainable_variables)
        
    def run(self, actor, critic, test=False):
        self.actor = actor
        self.critic = critic
        state = self.env.reset()
        
        # Agent has 500 trials at max, if it does not fail beforehand
        for _ in range(500):
            state = np.reshape(state, [1,8])
            if test: 
                self.env.render()
            # sample action from normal distribution
            action = self.get_action_distribution(state).sample()
            # Execute action and store action, state and reward 
            next_state, reward, done, _ = self.env.step(action[0])  # TODO Implement new storage system (array for each variable
            self.observations.append((state, action, reward))       # since the gradients are just accumulated and not applied after each iter)
            state = next_state
            # Interrupt the trial if the agent fails
            if done:
                break
            self.step += 1
        # Compute gradients in training
        if not test:
            return self._get_mean_gradients()
        else:
            for _, _, reward in self.observations:
                accum_reward =+ reward
            return accum_reward, self.step
            
    
    def _get_mean_gradients(self):
        # Lists to store gradients 
        policy_gradient_list = []
        critic_gradient_list = []
        # Iterate over taken actions and observed states and rewards
        self.observations.reverse()
        for state, action, reward in self.observations:
            # Compute estimated return
            self.estimated_return = self.gamma * self.estimated_return + reward
            # Compute gradients for the actor (policy gradient), Maximize the estimated return
            policy_gradient_list.append(self._compute_gradients('actor', state, action))
            # Compute gradients for the critic, minimize MSE for the state value function
            critic_gradient_list.append(self._compute_gradients('critic', state, action))
        # Compute mean gradients
        policy_gradients = np.mean(policy_gradient_list, axis=0)
        critic_gradients = np.mean(critic_gradient_list, axis=0)
        return policy_gradients, critic_gradients