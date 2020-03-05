import tensorflow.keras.layers as keras_layers
from tensorflow.keras.layers import Layer
import tensorflow as tf

# Value Fuction Estimator
class Critic(Layer):
    
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = keras_layers.Dense(units=128, input_shape=[8,], activation='relu')
        
        self.Dropout = keras_layers.Dropout(rate=0.2)

        self.fc2 = keras_layers.Dense(units=64, activation='relu')
        self.out = keras_layers.Dense(units=1, activation=None)
    
    def call(self, x):
        x = self.fc1(x)
        x = self.Dropout(x, training=True)
        x = self.fc2(x)
        x = self.out(x)
        return x

# Action Value Fuction Estimator (q-network)
class Actor(Layer):
    
    def __init__(self):
        super(Actor, self).__init__()
        
        # 64(share) -> 64(share) -> 32 -> 32 -> mu(tanh) [-1,1]
        # 64(share) -> 64(share) -> 32 -> 32 -> sigma(sigmoid) [0,1]
        self.sharedFC1 = keras_layers.Dense(units=64, input_shape=[8,], activation='relu')
        self.sharedFC2 = keras_layers.Dense(units=64, activation='relu')
        
        self.sharedBatchNorm = keras_layers.BatchNormalization()
        
        self.muFC1 = keras_layers.Dense(units=32, activation='relu')
        self.muFC2 = keras_layers.Dense(units=32, activation='relu')
        
        self.sigmaFC1 = keras_layers.Dense(units=32, activation='relu')
        self.sigmaFC2 = keras_layers.Dense(units=32, activation='relu')
        
        
        self.mu_out = keras_layers.Dense(units=2, activation='tanh')
        self.sigma_out = keras_layers.Dense(units=2, activation='sigmoid')
    
    def call(self, x):
        x = tf.convert_to_tensor(x)
        x = self.sharedFC1(x)
        x = self.sharedFC2(x)
        
        x = self.sharedBatchNorm(x, training=True)
        
        mu = self.muFC1(x)
        mu = self.muFC2(mu)
        mu = self.mu_out(mu)
        
        sigma = self.sigmaFC1(x)
        sigma = self.sigmaFC2(sigma)
        sigma = self.sigma_out(sigma)     
        
        return mu, sigma