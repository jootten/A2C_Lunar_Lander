from tensorflow_probability.python.distributions import Normal
import gym
from actor import Actor
import tensorflow as tf
import os
from datetime import datetime
import numpy as np
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def main():
    temp_env = gym.make('LunarLanderContinuous-v2')
    actor = Actor(temp_env)
    adam = tf.keras.optimizers.Adam()

    checkpoint_directory_a = "./training_checkpoints/actor"
    checkpoint = tf.train.Checkpoint(optimizer=adam, model=actor)
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory_a))

    env = gym.make('LunarLanderContinuous-v2')
    all_returns = 0
    rec_state = [None, None]
    for _ in range(200):    
        cum_return = 0 
        state = env.reset()
        while True:
            #env.render()
            state = np.reshape(state, (1,1,8))
            action_dist, rec_state = get_action_distribution(actor, state, rec_state) 
            action = action_dist.sample()[0]
            state, reward, done, _ = env.step(action)
            cum_return += reward
            if done:
                break
        print(cum_return)
        all_returns += cum_return
    print(cum_return / 200)

def get_action_distribution(actor, state, rec_state):
    mu, sigma, rec_state = actor(state, initial_state=rec_state)
    return Normal(loc=mu, scale=sigma), rec_state   

if __name__ == '__main__':
    main()
    