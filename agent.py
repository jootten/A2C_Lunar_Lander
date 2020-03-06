import tensorflow as tf
from tensorflow_probability.python.distributions import Normal
from actor import Actor
from critic import Critic
import gym
import numpy as np

class A2CAgent:
    def __init__(self):
        self.env = gym.make('LunarLanderContinuous-v2')
        
        self.gamma = 0.95 # discount factor
        self.max_time_steps = 500
        self.observations = []
        self.mse = tf.keras.losses.MSE
        self.accum_reward = 0.
        self.step = 0
        
    def run(self, actor, critic):
        state = self.env.reset()
        
        # Agent has 500 trials at max, if it does not fail beforehand
        for t in range(self.max_time_steps):
            self.env.render()
            # Compute action
            state = np.reshape(state, [1,8])
            mu, sigma = actor(state)

            # sample two values from normal distribution
            mainEngineAction = tf.random.normal((1,), mean=mu[0,0], stddev=sigma[0,0])
            sideEngineAction = tf.random.normal((1,), mean=mu[0,1], stddev=sigma[0,1])
            action = tf.concat([mainEngineAction, sideEngineAction], 0)

            # Execute action and store action, state and reward
            next_state, reward, done, info = self.env.step(action)
            self.observations.append((state, action, reward))
            state = next_state
            self.accum_reward += reward
            # Interrupt the trial if the agent fails
            if done:
                break
            self.step += 1

        # Initialize variable for the estimated return
        estimated_return = 0 if done else critic(next_state)

        # Iterate over taken actions and observed states and rewards
        self.observations.reverse()
        for state, action, reward in self.observations:
            # Compute estimated return
            estimated_return = self.gamma * estimated_return + reward
            # Compute state value
            state_v = critic(state)

            # Compute gradients for the actor (policy gradient)
            # Maximize the estimated return
            with tf.GradientTape() as actor_tape:
                mu, sigma = actor(state)
                advantages = estimated_return - int(state_v)
                advantages = tf.cast([[advantages]], tf.float32)
                
                action_distribution = Normal(loc=mu, scale=sigma)
                logprob = action_distribution.log_prob(action)
                actor_loss = logprob * advantages

                # Compute the actor loss (log part of the policy gradient)
                # Compute gradient with respect to the parameters of the actor            
                policy_gradients = actor_tape.gradient(actor_loss, actor.trainable_variables)

            # Compute gradients for the critic
            # minimize MSE for the state value function
            with tf.GradientTape() as critic_tape:
                state_v = critic(state)
                # Compute the loss
                critic_loss = self.mse(estimated_return, state_v)
                # Compute the gradient
                critic_gradients = critic_tape.gradient(critic_loss, critic.trainable_variables)
        
        return policy_gradients, critic_gradients

