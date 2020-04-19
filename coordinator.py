import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow_probability.python.distributions import Normal

import ray
import gym

from agent import A2CAgent
from memory import Memory

from actor import Actor
from critic import Critic

ENTROPY_COEF = 0.01 # used to balance exploration





class Coordinator:
    def __init__(self, num_agents=8, env_name='LunarLanderContinuous-v2', network='mlp', num_steps=32):
        # set up environment, observation memory 
        self.num_agents = num_agents
        self.num_steps=num_steps
        self.network = network
        
        temp_env = gym.make(env_name)
        self.obs_space_size = temp_env.observation_space.shape[0]
        
        self.memory = None
        
        # Initialize model, loss and optimizer
        self.actor = Actor(temp_env, network)
        self.critic = Critic()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.mse = tf.keras.losses.MeanSquaredError()
        self.actor_loss = None
        self.critic_loss = None

        # instantiate variable to store recurrent states of the agents
        self.agents_recurrent_state = None
        self.update_recurrent_state = None

        # store action distribution during update
        self.action_dist = None

        # Set up checkpoint paths
        self.checkpoint_directory_a = f"./training_checkpoints/{self.network}/actor"
        self.checkpoint_directory_c = f"./training_checkpoints/{self.network}/critic"
        
        # instantiate multiple agents (ray actors) and set first one as chief
        self.agent_list = [A2CAgent.remote(self.num_steps, env_name) for _ in range(num_agents)]
        self.agent_list[0].set_chief.remote()

        # Prepare Tensorboard
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        self.step = 0


    def train(self, num_updates=10000):
        # called from main
        cum_return = 0
        num_epsisodes = 0
        for i_update in range(num_updates):
            # Collect num_agents * num_steps observations
            for t in range(self.num_steps):
                memories = self.step_parallel(t)
            # Compute discounted return and concatenate memories from all agents
            [m.compute_discounted_cum_return(self.critic) for m in memories]
            self.memory = sum(memories)

            # calculate mean gradient over all agents and apply gradients to update models.
            mean_policy_gradients, mean_critic_gradients = self._get_mean_gradients()
            self.actor_optimizer.apply_gradients(zip(mean_policy_gradients, self.actor.trainable_variables))
            self.critic_optimizer.apply_gradients(zip(mean_critic_gradients, self.critic.trainable_variables))
            
            self.step += self.num_steps

            #store summary statistics
            with self.train_summary_writer.as_default():
                # Actor loss
                tf.summary.scalar('policy loss main engine', tf.reduce_mean(self.actor_loss[0], axis=0), step=self.step)
                tf.summary.scalar('policy loss side engines', tf.reduce_mean(self.actor_loss[1], axis=0), step=self.step)
                # Cumulative Return
                if True in self.memory.terminals[0:self.num_steps]:
                    idx = self.memory.terminals.index(True) + 1
                    cum_return += sum(self.memory.rewards[:idx,0])
                    tf.summary.scalar('cumulative return', cum_return, step=self.step)
                    cum_return = sum(self.memory.rewards[idx:self.num_steps,0])
                else:
                    cum_return += sum(self.memory.rewards[:self.num_steps,0])
                # Critic loss
                tf.summary.scalar('critic loss', self.critic_loss, step=self.step)

            # Store actor and critic model checkpoints
            if (i_update + 1) % 500 == 0:
                checkpoint_prefix_a = os.path.join(self.checkpoint_directory_a, f"{self.step}-{2}.ckpt")
                checkpoint_prefix_c = os.path.join(self.checkpoint_directory_c, f"{self.step}-{2}.ckpt")
                checkpoint = tf.train.Checkpoint(optimizer=self.actor_optimizer, model=self.actor)
                checkpoint.save(file_prefix=checkpoint_prefix_a)
                checkpoint = tf.train.Checkpoint(optimizer=self.critic_optimizer, model=self.critic)
                checkpoint.save(file_prefix=checkpoint_prefix_c)
            num_epsisodes += sum(self.memory.terminals)
            print(f"Update {i_update + 1} of {num_updates} finished with {self.num_agents} agents after {num_epsisodes} episodes.")



    def step_parallel(self, t):
        # Compute one step on all envs in parallel with gru as policy network
        if self.network == "gru":
            # Observe state to compute an action for the next time step
            states, dones = zip(*(ray.get([agent.observe.remote(t) for agent in self.agent_list])))
            # create mask to reset the  recurrent state for finished environments
            mask = np.ones((self.num_agents, 1, self.obs_space_size))
            if True in dones:
                mask[dones,:,:] = 0
            input = tf.concat((np.array(states), mask), axis=2)
            # Sample action from the normal distribution given by the policy
            action_dist, self.agents_recurrent_state = self.get_action_distribution(input, recurrent_state=self.agents_recurrent_state)
            actions = np.array(action_dist.sample())

        # Compute one step on all envs in parallel with mlp as policy network
        if self.network == "mlp":
            # Observe state to compute an action for the next time step
            states, dones = zip(*(ray.get([agent.observe.remote(t) for agent in self.agent_list])))
            action_dist, _ = self.get_action_distribution(np.array(states))
            # Sample action from the normal distribution given by the policy
            actions = np.array(action_dist.sample())

        # Execute action and obtain memory after num_steps
        memories = ray.get([agent.execute.remote(actions[i], t) for i, agent in enumerate(self.agent_list)])
        return memories

    def get_action_distribution(self, state, recurrent_state=None, update=False):
        # Get the normal distribution over the action space, determined by mu and sigma
        if self.network == "mlp":
            mu, sigma, _ = self.actor(state.squeeze())
            return Normal(loc=mu, scale=sigma), None

        if self.network == "gru":
            if update:
                # create mask to reset the  recurrent state for finished environments
                mask = np.ones(((self.num_agents * self.num_steps), self.obs_space_size))
                mask[self.memory.terminals,:] = 0
                state = tf.concat((state, mask), axis=1)
                state = tf.reshape(state, (self.num_agents, self.num_steps, (self.obs_space_size * 2)))
            # forward mask together with input to the actor
            mu, sigma, recurrent_state = self.actor(state, initial_state=recurrent_state)
            return Normal(loc=mu, scale=sigma), recurrent_state
        
    def _get_mean_gradients(self):
        # Compute gradients for the actor (policy gradient), Maximize the estimated return
        self.actor_loss, policy_gradients = self._compute_gradients('actor')
        # Compute gradients for the critic, minimize MSE for the state value function
        self.critic_loss, critic_gradients = self._compute_gradients('critic')
        return policy_gradients, critic_gradients

    def _compute_gradients(self, type):
        with tf.GradientTape() as tape:
            if type == 'actor':
                # Compute the actor loss: 
                loss = self._actor_loss() - self.action_dist.entropy() * ENTROPY_COEF
            else:
                # Compute the state value
                state_v = self.critic(self.memory.states, training=True)
                # Compute the critic loss
                loss = self.mse(self.memory.estimated_return, state_v, sample_weight=0.5)
            # Compute the gradients
            return loss, tape.gradient(loss, self.actor.trainable_variables if type == 'actor' else self.critic.trainable_variables)
    
    def _actor_loss(self):
        # Compute state value
        state_v = self.critic(self.memory.states)
        # Get advantage estimate
        advantages = self.memory.estimated_return - state_v
        advantages = tf.cast(advantages, tf.float32)
        # Get log probability of the taken action
        self.action_dist, self.update_recurrent_state = self.get_action_distribution(self.memory.states, update=True)
        logprob = self.action_dist.log_prob(self.memory.actions)
        # Advantage as baseline
        return -logprob * advantages
