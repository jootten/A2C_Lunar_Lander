import tensorflow.keras.layers as keras_layers
from tensorflow.keras.layers import Layer
import tensorflow as tf

# Action Value Fuction Estimator (q-network)
class Actor(Layer):
    
    def __init__(self, env, network):
        super(Actor, self).__init__()
        
        self.action_space_size = 1 if env.action_space.shape == () else env.action_space.shape[0]
        self.type = network

        if self.type == "lstm":
            self.lstm1 = keras_layers.LSTM(32, return_sequences=True, return_state=True)
            self.lstm2 = keras_layers.LSTM(32, return_sequences=True, return_state=True)

        if self.type == "mlp":
            self.fc1 = keras_layers.Dense(units=128, activation='relu', kernel_regularizer="l2")
            self.fc2 = keras_layers.Dense(units=64, activation='relu')
            self.fc3 = keras_layers.Dense(units=32, activation='relu')
        
        self.batch_norm = keras_layers.BatchNormalization()
        
        self.mu_out = keras_layers.Dense(units=self.action_space_size, activation='tanh')
        self.sigma_out = keras_layers.Dense(units=self.action_space_size, activation='softplus')
    
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
            #x = self.batch_norm(x)
            x = self.fc3(x)

        mu = self.mu_out(x)
        sigma = self.sigma_out(x)     
        
        return tf.reshape(mu, [-1, 2]), tf.reshape(sigma, [-1, 2]), state
