from agent import A2CAgent
from actor import Actor
from critic import Critic
import tensorflow as tf
import os
from datetime import datetime

os.system("rm -rf ./logs/")

#!rm -rf ./logs/

class Coordinator:
    def __init__(self, num_agents=1, num_episodes=100):
        self.num_agents = num_agents
        self.agent_list = []
        
        # Initialize model, loss and optimizer
        self.actor = Actor()
        self.critic = Critic()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
        
        # create multiple agents
        for i_agent in range(num_agents):
            self.agent_list.append(A2CAgent())

        # Prepare Tensorboard
        
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)


    def run_for_episodes(self, num_episodes=100):
        # called from main
        
        for i_episode in range(num_episodes):
            # gradient lists to collect gradients from all agents
            policy_gradient_list = []
            critic_gradient_list = []
            for agent in self.agent_list:
                policy_gradient, critc_gradient = agent.run(self.actor, self.critic)
                policy_gradient_list.append(policy_gradient)
                critic_gradient_list.append(critc_gradient)
                print(f"Episode {i_episode + 1} of {num_episodes} finished")
                
        # calculate mean gradient over all agents and apply gradients to update models.
        mean_policy_gradients = tf.math.reduce_mean(policy_gradient_list)
        mean_critic_gradients = tf.math.reduce_mean(critic_gradient_list)
        actor_optimizer.apply_gradients(zip(mean_policy_gradients, actor.trainable_variables))
        critic_optimizer.apply_gradients(zip(mean_critic_gradients, critic.trainable_variables))

        # Store summary statistics
        with self.train_summary_writer.as_default():
            tf.summary.scalar('policy loss', tf.reduce_mean(actor_losses), step=step)
            
            # Store summary statistics
            tf.summary.scalar('critic loss', tf.reduce_mean(critic_losses), step=step)
            
            # Critic
            #tf.summary.scalar('V(s)', state_v[0,0], step=step)
            
            # Actor
            tf.summary.scalar('mu0', mu[0,0], step=step)
            tf.summary.scalar('sigma0', sigma[0,0], step=step)
            tf.summary.scalar('mu1', mu[0,1], step=step)
            tf.summary.scalar('sigma1', sigma[0,1], step=step)
            
            # Accumulative reward
            tf.summary.scalar("accumulative reward", accum_reward, step=step)