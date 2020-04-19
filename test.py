from tensorflow_probability.python.distributions import Normal
import gym
from actor import Actor
import tensorflow as tf
import os
from datetime import datetime
import numpy as np

def test_run(network="mlp", environment='LunarLanderContinuous-v2'):
    # Initialize model and environment
    env = gym.make(environment)
    actor = Actor(env, network)

    # Load model checkpoint
    checkpoint_directory_a = f"./training_checkpoints/{network}/actor"
    checkpoint = tf.train.Checkpoint(model=actor)
    _ = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory_a))

    all_returns = 0
    rec_state = None
    # Run the agent for 100 consecutive trials
    for episode in range(100):    
        cum_return = 0 
        state = env.reset()
        while True:
            env.render()
            # Get action
            if network == "gru":
                # create mask to reset the  recurrent state for finished environments
                mask = np.ones((1, 1, env.observation_space.shape[0]))
                input = tf.concat((state.reshape(1,1,8), mask), axis=2)
                # Sample action from the normal distribution given by the policy
                action_dist, rec_state = get_action_distribution(actor, input, network, recurrent_state=rec_state)
                action = np.array(action_dist.sample())

            if network == "mlp":
                state = np.reshape(state, (1,8))
                action_dist, _ = get_action_distribution(actor, state, network)
            action = action_dist.sample()[0]
            # Execute action and accumulate rewards
            state, reward, done, _ = env.step(action)
            cum_return += reward
            if done:
                rec_state = None
                break
        print(f"Total reward in episode {episode}: {cum_return}.")
        all_returns += cum_return
    print(f"Average cumulative return after 200 episodes: {all_returns / 100}.")

# Compute action distribution with trained actor/policy network
def get_action_distribution(actor, state, network, recurrent_state=None):
    if network == "gru":
        mu, sigma, recurrent_state = actor(state, initial_state=recurrent_state)
        return Normal(loc=mu, scale=sigma), recurrent_state
    
    if network == "mlp":
        mu, sigma, _ = actor(state)
        return Normal(loc=mu, scale=sigma), None