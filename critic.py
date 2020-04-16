import tensorflow.keras.layers as keras_layers
from tensorflow.keras.layers import Layer
import tensorflow as tf

# State value fuction estimator used to compute the advantage
class Critic(Layer):
    
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = keras_layers.Dense(units=128, input_shape=[8,], activation='relu', kernel_regularizer="l2")
        self.fc2 = keras_layers.Dense(units=64, activation='relu')
        self.out = keras_layers.Dense(units=1, activation=None)
    
    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x

