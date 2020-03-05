import tensorflow as tf
from tensorflow_probability.python.distributions import Normal

class A2CAgent:
    def __init__(self):
        self.gamma = 0.95

class Coordinator:
    def __init__(self, num_agents=1):
        self.num_agents = num_agents