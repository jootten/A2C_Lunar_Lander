import ray
import gym
import numpy as np
from parallel_test_actor import Actor

@ray.remote    
def setup(): 
    env = gym.make('LunarLanderContinuous-v2')
    env.reset()
    
    observations = []
    
    for _ in range(1000):
        env.render()
        state, reward, done, info = env.step(env.action_space.sample())
        
        observations.append((state, reward))
        
        if done:
                break
    env.close()
    
    return observations
    
def main():
    ray.init()
    
    # parallelize classes
    agents = [Actor.remote() for _ in range(2)]
    #a1 = Actor.remote()
    #a2 = Actor.remote()
    
    obs_ids = ray.get([agent.run1ep.remote() for agent in agents])
    #obs1_id = a1.run1ep.remote()
    #obs2_id = a2.run1ep.remote()
    
    # Get returned observations with ray.get(obs1_id)
    
    # Or parallelize functions: 
    #obs_id1 = setup.remote()
    #obs_id2 = setup.remote()

    breakpoint()

    ray.shutdown()

    breakpoint()
    
if __name__ == '__main__':
    main()