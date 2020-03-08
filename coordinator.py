from agent import A2CAgent
from actor import Actor
from agent import Memory
from critic import Critic
import tensorflow as tf
import os
from datetime import datetime
import numpy as np

os.system("rm -rf ./logs/")

#!rm -rf ./logs/

class Coordinator:
    def __init__(self, num_agents=8, num_updates=10000):
        self.num_agents = num_agents
        self.num_steps = 8
        self.agent_list = []
        
        # Initialize model, loss and optimizer
        self.actor = Actor()
        self.critic = Critic()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
        
        # create multiple agents
        for _ in range(num_agents):
            self.agent_list.append(A2CAgent())

        # Prepare Tensorboard
        
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)


    def run_for_episodes(self, num_updates=10000):
        # called from main
        
        for i_update in range(num_updates):
            # gradient lists to collect gradients from all agents
            policy_gradient_list = []
            critic_gradient_list = []
            for agent in self.agent_list:
                policy_gradient, critic_gradient = agent.run(self.actor, self.critic, self.num_steps)
                policy_gradient_list.append(policy_gradient)
                critic_gradient_list.append(critic_gradient)
                
                
            # calculate mean gradient over all agents and apply gradients to update models.
            mean_policy_gradients = np.mean(policy_gradient_list, axis=0)
            mean_critic_gradients = np.mean(critic_gradient_list, axis=0)
            self.actor_optimizer.apply_gradients(zip(mean_policy_gradients, self.actor.trainable_variables))
            self.critic_optimizer.apply_gradients(zip(mean_critic_gradients, self.critic.trainable_variables))
            # 
            print(f"Update {i_update + 1} of {num_updates} with {self.num_agents} agents finished.")
            # Render environment and store summary statistics
            if i_update % 20 == 0:
                self.test()

    def test(self):
        self.agent_list[0].run(self.actor, self.critic, test=True, num_steps=200)
        accum_reward = self.agent_list[0].memory.rewards.sum()
        step = self.agent_list[0].step
        # Store summary statistics
        with self.train_summary_writer.as_default():
            #tf.summary.scalar('policy loss', tf.reduce_mean(actor_losses), step=step)
            
            # Store summary statistics
            #tf.summary.scalar('critic loss', tf.reduce_mean(critic_losses), step=step)
            
            # Critic
            tf.summary.scalar('accumulative reward', accum_reward, step=step)
            
            # Actor
            #tf.summary.scalar('mu0', dist.loc[0,0], step=step)
            #tf.summary.scalar('sigma0', dist.scale[0,1], step=step)
            #tf.summary.scalar('mu1', mu[0,1], step=step)
            #tf.summary.scalar('sigma1', sigma[0,1], step=step)
            
            # Accumulative reward
            #tf.summary.scalar("accumulative reward", accum_reward, step=step)