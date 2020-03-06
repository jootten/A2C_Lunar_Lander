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

