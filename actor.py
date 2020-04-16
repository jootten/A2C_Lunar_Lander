import tensorflow.keras.layers as kl
from tensorflow.keras.layers import Layer
import tensorflow as tf

# Policy/actor network
# Estimates the parameters mu and sigma of the normal distribution
# used to sample the actions for the agent
class Actor(Layer):
    
    def __init__(self, env, network):
        super(Actor, self).__init__()
        
        self.action_space_size = env.action_space.shape[0]
        self.type = network

        if self.type == "lstm":
            self.lstm1 = kl.LSTM(32, return_sequences=True, return_state=True)
            self.lstm2 = kl.LSTM(32, return_sequences=True, return_state=True)

        if self.type == "mlp":
            self.fc1 = kl.Dense(units=128, activation='relu', kernel_regularizer="l2")
            self.fc2 = kl.Dense(units=64, activation='relu')
            self.fc3 = kl.Dense(units=32, activation='relu')
        
        self.batch_norm = kl.BatchNormalization()
        
        self.mu_out = kl.Dense(units=self.action_space_size, activation='tanh')
        self.sigma_out = kl.Dense(units=self.action_space_size, activation='softplus')
    
    def call(self, x, initial_state=[None, None]):
        state = None

        if self.type == "lstm":
            x, s1_h, s1_c = self.lstm1(x, initial_state=initial_state[0])
            x = self.batch_norm(x)
            x, s2_h, s2_c = self.lstm2(x, initial_state=initial_state[1])
            
            state_1 = [s1_h, s1_c]
            state_2 = [s2_h, s2_c]
            state = [state_1, state_2]

        if self.type == "mlp":
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)

        mu = self.mu_out(x)
        sigma = self.sigma_out(x)     
        
        return tf.reshape(mu, [-1, 2]), tf.reshape(sigma, [-1, 2]), state
