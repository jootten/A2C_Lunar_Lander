from agent import A2CAgent
from actor import Actor
from memory import Memory
from critic import Critic
import tensorflow as tf
import os
from datetime import datetime
import numpy as np
import ray
from tensorflow_probability.python.distributions import Normal

os.system("rm -rf ./logs/")

#!rm -rf ./logs/

class Coordinator:
    def __init__(self, num_agents=4):
        self.num_agents = num_agents
        self.num_steps = 5
        self.agent_list = []
        self.memory = Memory(self.num_steps)
        self.batch_size = self.num_steps * self.num_agents
        self.estimated_return = np.zeros(shape=(self.batch_size, 1))
        
        # Initialize model, loss and optimizer
        self.actor = Actor()
        self.critic = Critic()
        self.mse = tf.keras.losses.MeanSquaredError()
        self.gamma = 0.99
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
        
        self.entropy_coefficient = 0.01 # used to balance exploration
        
        # create multiple agents
#        for _ in range(num_agents):
        self.agent_list = [A2CAgent.remote() for _ in range(num_agents)]

        # Prepare Tensorboard
        
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)


    def run_for_episodes(self, num_updates=10000):
        # called from main
        
        for i_update in range(num_updates):
            memories = ray.get([agent.run.remote(self.actor, self.critic, self.num_steps) for agent in self.agent_list])
            self.memory = sum(memories)
            print(f"Episode {i_update + 1} of {num_updates} finished")
                
            # calculate mean gradient over all agents and apply gradients to update models.
            mean_policy_gradients, mean_critic_gradients = self._get_mean_gradients()
            self.actor_optimizer.apply_gradients(zip(mean_policy_gradients, self.actor.trainable_variables))
            self.critic_optimizer.apply_gradients(zip(mean_critic_gradients, self.critic.trainable_variables))
            # Render environment and store summary statistics
            if i_update % 50 == 0:
                self.test()

    def test(self):
        pass
        self.agent_list[0].run.remote(self.actor, self.critic, test=True, num_steps=200)
        #$accum_reward = self.agent_list[0].memory.rewards.sum()
        #step = self.agent_list[0].step
        # Store summary statistics
        #with self.train_summary_writer.as_default():
            #tf.summary.scalar('policy loss', tf.reduce_mean(actor_losses), step=step)
            
            # Store summary statistics
            #tf.summary.scalar('critic loss', tf.reduce_mean(critic_losses), step=step)
            
            # Critic
            #tf.summary.scalar('accumulative reward', accum_reward, step=step)
            
            # Actor
            #tf.summary.scalar('mu0', mu[0,0], step=step)
            #tf.summary.scalar('sigma0', sigma[0,0], step=step)
            #tf.summary.scalar('mu1', mu[0,1], step=step)
            #tf.summary.scalar('sigma1', sigma[0,1], step=step)
            
            # Accumulative reward
            #tf.summary.scalar("accumulative reward", accum_reward, step=step)
    def _entropy(self):
        # get standard deviation values
        std = self.get_action_distribution(self.memory.states).scale
        logstd = np.log(std)
        
        # use log std to calculate entropy
        entropy = tf.reduce_sum(logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)
        # expand dimensions t
        entropy = tf.expand_dims(entropy,1)
        return tf.reduce_mean(entropy) #tf.zeros((1,20))
        
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
                # Compute the actor loss: 
                # total loss = policy gradient loss - entropy * entropy coefficient +  value loss * Value coefficient 
                loss = self._actor_loss() - self._entropy() * self.entropy_coefficient #+ value_loss * value_coefficient ? 
                #breakpoint()
            else:
                # Compute the statue value
                state_v = self.critic(self.memory.states)
                # Compute the critic loss
                loss = self.mse(self.estimated_return, state_v)
            # Compute the gradients
            return tape.gradient(loss, self.actor.trainable_variables if type == 'actor' else self.critic.trainable_variables)

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

    def get_action_distribution(self, state):
        mu, sigma = self.actor(state)
        return Normal(loc=mu, scale=sigma)