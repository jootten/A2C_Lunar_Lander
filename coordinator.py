import os
from agent import A2CAgent
from actor import Actor
from memory import Memory
from critic import Critic
import tensorflow as tf
from datetime import datetime
import numpy as np
import ray
import gym
from tensorflow_probability.python.distributions import Normal

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

print(physical_devices)

ENV = 'LunarLanderContinuous-v2'

os.system("rm -rf ./logs/")

ENTROPY_COEFFICIENT = 0.001
NUM_STEPS = 16

class Coordinator:
    def __init__(self, num_agents=8):
        self.num_agents = num_agents
        self.agent_list = []
        self.memory = None
        self.step = 0
        self.actor_loss = None
        self.critic_loss = None
        self.previous_recurrent_state = [None, None]
        
        # Initialize model, loss and optimizer
        self.actor = Actor()
        self.critic = Critic()
        self.mse = tf.keras.losses.MeanSquaredError()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        self.checkpoint_directory_a = "./training_checkpoints/actor"
        self.checkpoint_directory_c = "./training_checkpoints/critic"

        self.entropy_coefficient = 0.001 # used to balance exploration
        
        # create multiple agents
        self.agent_list = [A2CAgent.remote(NUM_STEPS) for _ in range(num_agents)]

        # Prepare Tensorboard
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)


    def run_for_episodes(self, num_updates=500000):
        # called from main
        for i_update in range(num_updates):
            memories = ray.get([agent.run.remote(self.actor, self.critic) for agent in self.agent_list])
            self.memory = sum(memories)
            del memories
            # calculate mean gradient over all agents and apply gradients to update models.
            mean_policy_gradients, mean_critic_gradients = self._get_mean_gradients()
            self.actor_optimizer.apply_gradients(zip(mean_policy_gradients, self.actor.trainable_variables))
            self.critic_optimizer.apply_gradients(zip(mean_critic_gradients, self.critic.trainable_variables))
            
            self.step += NUM_STEPS

            #if i_update % 8 == 0:
            #    self.test()
            #store summary statistics
            with self.train_summary_writer.as_default():
                # Actor loss
                tf.summary.scalar('policy loss main engine', tf.reduce_mean(self.actor_loss[0], axis=0), step=self.step)
                tf.summary.scalar('policy loss side engines', tf.reduce_mean(self.actor_loss[1], axis=0), step=self.step)
                
                # Critic loss
                tf.summary.scalar('critic loss', self.critic_loss, step=self.step)
            if i_update % 1000 == 0:
                checkpoint_prefix_a = os.path.join(self.checkpoint_directory_a, f"{self.step}-{2}.ckpt")
                checkpoint_prefix_c = os.path.join(self.checkpoint_directory_c, f"{self.step}-{2}.ckpt")
                checkpoint = tf.train.Checkpoint(optimizer=self.actor_optimizer, model=self.actor)
                checkpoint.save(file_prefix=checkpoint_prefix_a)
                checkpoint = tf.train.Checkpoint(optimizer=self.critic_optimizer, model=self.critic)
                checkpoint.save(file_prefix=checkpoint_prefix_c)
            print(f"Update {i_update + 1} of {num_updates} finished with {self.num_agents} agents.")




            

    def _entropy(self):
        # get standard deviation values
        std = self.get_action_distribution(self.memory.states).scale
        logstd = np.log(std)
        # use log std to calculate entropy
        entropy = tf.reduce_sum(logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)
        # expand dimensions t
        entropy = tf.expand_dims(entropy,1)
        return tf.reduce_mean(entropy)
        
    def _actor_loss(self):
        # Compute state value
        state_v = self.critic(self.memory.states)
        # Get advantage estimate
        advantages = self.memory.estimated_return - state_v
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
            else:
                # Compute the statue value
                state_v = self.critic(self.memory.states, training=True)
                # Compute the critic loss
                loss = self.mse(self.memory.estimated_return, state_v)
            # Compute the gradients
            return loss, tape.gradient(loss, self.actor.trainable_variables if type == 'actor' else self.critic.trainable_variables)

    def _get_mean_gradients(self):
        # Compute gradients for the actor (policy gradient), Maximize the estimated return
        self.actor_loss, policy_gradients = self._compute_gradients('actor')
        # Compute gradients for the critic, minimize MSE for the state value function
        self.critic_loss, critic_gradients = self._compute_gradients('critic')
        return policy_gradients, critic_gradients

    def get_action_distribution(self, state):
        mu, sigma, self.previous_recurrent_state = self.actor(state.reshape(self.num_agents, NUM_STEPS, 8), initial_state=self.previous_recurrent_state)
        return Normal(loc=mu, scale=sigma)

    def test(self):
        memory = ray.get(self.agent_list[0].run.remote(self.actor, self.critic, test=True, num_steps=200))
        accum_reward = memory.rewards.sum()
        # Store summary statistics
        with self.train_summary_writer.as_default():
            # Accumulative reward
            tf.summary.scalar('accumulative reward', accum_reward, step=self.step)


