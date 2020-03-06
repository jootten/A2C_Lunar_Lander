from agent import A2CAgent
from actor import Actor
from critic import Critic
import tensorflow as tf

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

    def run_for_episodes(self, num_episodes=100):
        # called from main
        
        for i_episode in range(num_episodes):
            for agent in self.agent_list:
                agent.run(self.actor, self.critic)
                print(f"Episode {i_episode + 1} of {num_episodes} finished")
                
