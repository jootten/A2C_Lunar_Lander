from tensorflow_probability.python.distributions import Normal
import gym
from actor import Actor
import tensorflow as tf
import os
from datetime import datetime
import numpy as np
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def test_run(network="mlp", environment='LunarLanderContinuous-v2'):
    # Initialize model and environment
    env = gym.make(environment)
    actor = Actor(temp_env, network="mlp")
    adam = tf.keras.optimizers.Adam()

    # Load model checkpoint
    checkpoint_directory_a = f"./training_checkpoints/{network}/actor"
    checkpoint = tf.train.Checkpoint(optimizer=adam, model=actor)
    _ = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory_a))

    all_returns = 0
    rec_state = [None, None]
    # Run the agent for 100 consecutive trials
    for episode in range(100):    
        cum_return = 0 
        state = env.reset()
        while True:
            env.render()
            # Get action
            if network == "lstm":
                state = np.reshape(state, (1,1,8))
                action_dist, rec_state = get_action_distribution(actor, state, network, rec_state) 
            if network == "mlp":
                state = np.reshape(state, (1,8))
                action_dist, _ = get_action_distribution(actor, state, network)
            action = action_dist.sample()[0]
            # Execute action and accumulate rewards
            state, reward, done, _ = env.step(action)
            cum_return += reward
            if done:
                break
        print(f"Total reward in episode {episode}: {cum_return}.")
        all_returns += cum_return
    print(f"Average cumulative return after 200 episodes: {all_returns / 100}.")

# Compute action distribution with trained actor/policy network
def get_action_distribution(actor, state, network, recurrent_state=[None, None]):
    if network == "lstm":
        mu, sigma, recurrent_state = actor(state, initial_state=recurrent_state)
        return Normal(loc=mu, scale=sigma), recurrent_state
    
    if network == "mlp":
        mu, sigma, _ = actor(state)
        return Normal(loc=mu, scale=sigma), None