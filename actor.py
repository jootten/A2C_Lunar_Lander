import tensorflow.keras.layers as kl
from tensorflow.keras.layers import Layer
import tensorflow as tf
from gru import GRUCell

# Policy/actor network
# Estimates the parameters mu and sigma of the normal distribution
# used to sample the actions for the agent
class Actor(Layer):
    
    def __init__(self, env, network):
        super(Actor, self).__init__()
        
        self.action_space_size = env.action_space.shape[0]
        self.observation_space_size = env.observation_space.shape[0]
        self.type = network

        if self.type == "gru":
            self.cell_1 = GRUCell(input_dim=self.observation_space_size, units=64)
            self.cell_2 = GRUCell(input_dim=64, units=32)
            self.fc_gru = kl.Dense(units=32)
            self.cells = [self.cell_1, self.cell_2]
            self.rnn = tf.keras.layers.RNN(self.cells, return_sequences=True, return_state=True)

        if self.type == "mlp":
            self.fc1 = kl.Dense(units=128, activation='relu', kernel_regularizer="l2")
            self.fc2 = kl.Dense(units=64, activation='relu')
            self.fc3 = kl.Dense(units=32, activation='relu')
        
        self.batch_norm = kl.BatchNormalization()
        
        self.mu_out = kl.Dense(units=self.action_space_size, activation='tanh')
        self.sigma_out = kl.Dense(units=self.action_space_size, activation='softplus')
        
    def call(self, x, initial_state=None):
        state = None

        if self.type == "gru":
            x, state_1, state_2 = self.rnn(x, initial_state=initial_state)
            x, _ = tf.split(x, 2, axis=2)
            state = [state_1, state_2]
            x = self.fc_gru(x)

        if self.type == "mlp":
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)

        mu = self.mu_out(x)
        sigma = self.sigma_out(x)     
        
        return tf.reshape(mu, [-1, self.action_space_size]), tf.reshape(sigma, [-1, self.action_space_size]), state
