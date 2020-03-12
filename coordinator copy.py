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

ENTROPY_COEFFICIENT = 0.001
NUM_STEPS = 128

class Coordinator:
    def __init__(self, num_agents=4):
        self.num_agents = num_agents
        self.agent_list = []
        self.batch_size = NUM_STEPS * self.num_agents
        self.step = 0
        self.actor_loss = None
        self.critic_loss = None
        
        # Initialize model, loss and optimizer
        self.actor = Actor()
        self.critic = Critic()
        self.mse = tf.keras.losses.MeanSquaredError()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
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
            memories = ray.get([agent.run.remote(self.actor, self.critic, NUM_STEPS) for agent in self.agent_list])
            # calculate mean gradient over all agents and apply gradients to update models.
            mean_policy_gradients, mean_critic_gradients = self._get_mean_gradients(memories)
            self.actor_optimizer.apply_gradients(zip(mean_policy_gradients, self.actor.trainable_variables))
            self.critic_optimizer.apply_gradients(zip(mean_critic_gradients, self.critic.trainable_variables))
            # Render environment
            self.step += NUM_STEPS

            if i_update % 8 == 0:
                self.test()
            #store summary statistics
            with self.train_summary_writer.as_default():
                # Actor loss
                tf.summary.scalar('policy loss main engine', tf.reduce_mean(self.actor_loss[0], axis=0), step=self.step)
                tf.summary.scalar('policy loss side engines', tf.reduce_mean(self.actor_loss[1], axis=0), step=self.step)
                
                # Critic loss
                tf.summary.scalar('critic loss', self.critic_loss, step=self.step)
            if i_update % 1000 == 0:
                checkpoint_directory_a = "./training_checkpoints/actor"
                checkpoint_directory_c = "./training_checkpoints/critic"
                checkpoint_prefix_a = os.path.join(checkpoint_directory_a, f"{self.step}.ckpt")
                checkpoint_prefix_c = os.path.join(checkpoint_directory_c, f"{self.step}.ckpt")
                checkpoint = tf.train.Checkpoint(optimizer=self.actor_optimizer, model=self.actor)
                checkpoint.save(file_prefix=checkpoint_prefix_a)
                checkpoint = tf.train.Checkpoint(optimizer=self.critic_optimizer, model=self.critic)
                checkpoint.save(file_prefix=checkpoint_prefix_c)
            print(f"Update {i_update + 1} of {num_updates} finished with {self.num_agents} agents.")




            

    def _entropy(self, memory):
        # get standard deviation values
        std = self.get_action_distribution(memory.states).scale
        logstd = np.log(std)
        # use log std to calculate entropy
        entropy = tf.reduce_sum(logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)
        # expand dimensions t
        entropy = tf.expand_dims(entropy,1)
        return tf.reduce_mean(entropy)
        
    def _actor_loss(self, memory):
        # Compute state value
        state_v = self.critic(memory.states)
        # Get advantage estimate
        advantages = memory.estimated_return - state_v
        advantages = tf.cast(advantages, tf.float32)
        # Get log probability of the taken action
        logprob = self.get_action_distribution(memory.states).log_prob(memory.actions)
        # Advantage as baseline
        return -logprob * advantages

    def _compute_gradients(self, type, memory):
        with tf.GradientTape() as tape:
            if type == 'actor':
                # Compute the actor loss: 
                # total loss = policy gradient loss - entropy * entropy coefficient +  value loss * Value coefficient 
                loss = self._actor_loss(memory) - self._entropy(memory) * self.entropy_coefficient #+ value_loss * value_coefficient ? 
                #breakpoint()
            else:
                # Compute the statue value
                state_v = self.critic(memory.states)
                # Compute the critic loss
                loss = self.mse(memory.estimated_return, state_v)
            # Compute the gradients
            return loss, tape.gradient(loss, self.actor.trainable_variables if type == 'actor' else self.critic.trainable_variables)

    def _get_mean_gradients(self, memories):
        polily_gradients_list = []
        critic_gradients_list = []
        for memory in memories:
            # Compute gradients for the actor (policy gradient), Maximize the estimated return
            actor_loss, policy_gradients = self._compute_gradients('actor', memory)
            
            # Compute gradients for the critic, minimize MSE for the state value function
            self.critic_loss, critic_gradients = self._compute_gradients('critic', memory)
        return policy_gradients / self.num_agents, critic_gradients / self.num_agents

    def get_action_distribution(self, state):
        mu, sigma = self.actor(state)
        return Normal(loc=mu, scale=sigma)

    def test(self):
        memory = ray.get(self.agent_list[0].run.remote(self.actor, self.critic, test=True, num_steps=200))
        accum_reward = memory.rewards.sum()
        # Store summary statistics
        with self.train_summary_writer.as_default():
            # Accumulative reward
            tf.summary.scalar('accumulative reward', accum_reward, step=self.step)

            
            