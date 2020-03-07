import ray
import gym

@ray.remote
class Actor():
    def __init__(self):
        self.env = gym.make('LunarLanderContinuous-v2')
        self.env.reset()
        self.observations = []
        
    def run1ep(self):
        for _ in range(1000):
            self.env.render()
            state, reward, done, info = self.env.step(self.env.action_space.sample())

            self.observations.append((state, reward))

            if done:
                    break
        self.env.close()

        return self.observations
