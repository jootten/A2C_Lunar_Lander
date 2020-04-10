import tensorflow.keras.layers as keras_layers
from tensorflow.keras.layers import Layer
import tensorflow as tf

# Action Value Fuction Estimator (q-network)
class Actor(Layer):
    
    def __init__(self, env):
        super(Actor, self).__init__()
 
        self.lstm1 = keras_layers.LSTM(64, return_sequences=True, return_state=True)
        
        self.action_space_size = 1 if env.action_space.shape == () else env.action_space.shape[0]

        self.batch_norm = keras_layers.BatchNormalization()
        
        self.lstm2 = keras_layers.LSTM(32, return_sequences=True, return_state=True)
        
        self.mu_out = keras_layers.Dense(units=self.action_space_size, activation='tanh')
        self.sigma_out = keras_layers.Dense(units=self.action_space_size, activation='softplus')
    
    def call(self, x, initial_state=[None, None]):
        x, s1_h, s1_c = self.lstm1(x, initial_state=initial_state[0])
        x = self.batch_norm(x)
        x, s2_h, s2_c = self.lstm2(x, initial_state=initial_state[1])
        
        state_1 = [s1_h, s1_c]
        state_2 = [s2_h, s2_c]
        state = [state_1, state_2]

        mu = self.mu_out(x)
        sigma = self.sigma_out(x)     
        
        return tf.reshape(mu, [-1, 2]), tf.reshape(sigma, [-1, 2]), state