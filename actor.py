import tensorflow.keras.layers as keras_layers
from tensorflow.keras.layers import Layer
import tensorflow as tf

# Action Value Fuction Estimator (q-network)
class Actor(Layer):
    
    def __init__(self, env):
        super(Actor, self).__init__()
        
        self.action_space_size = 1 if env.action_space.shape == () else env.action_space.shape[0]

        self.lstm1 = keras_layers.LSTM(64, return_sequences=True, stateful=True)
        self.lstm2 = keras_layers.LSTM(32, return_sequences=True, stateful=True)

        # 64(share) -> 64(share) -> 32 -> 32 -> mu(tanh) [-1,1]
        # 64(share) -> 64(share) -> 32 -> 32 -> sigma(sigmoid) [0,1]
        self.sharedFC1 = keras_layers.Dense(units=128, activation='relu')
        self.sharedFC2 = keras_layers.Dense(units=64, activation='relu')
        
        self.sharedBatchNorm = keras_layers.BatchNormalization()
        
        self.muFC1 = keras_layers.Dense(units=32, activation='relu')
        
        self.mu_out = keras_layers.Dense(units=self.action_space_size, activation='tanh')
        self.sigma_out = keras_layers.Dense(units=self.action_space_size, activation='softplus')
    
    def call(self, x):
        # x = tf.convert_to_tensor(x)
        # x = self.sharedFC1(x)
        # x = self.sharedFC2(x)
        # x = self.sharedBatchNorm(x)

        # x = self.muFC1(x)

        x = self.lstm1(x)
        x = self.lstm2(x)

        mu = self.mu_out(x)
        sigma = self.sigma_out(x)     
        
        return mu, sigma

    def reset_states(self):
        self.lstm1.states = [None,None]
        self.lstm2.states = [None,None]