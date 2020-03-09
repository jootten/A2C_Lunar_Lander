## Project report

#### topic:

* A2C for continuous action spaces applied on the [LunarLanderContinuous][LLC] environment from OpenAI

#### important parts we worked on:

**Phase 1** (implementing A2C for the discrete action space in the [CartPole][CP] environment:

* getting the gym environment to run
* setting up two simple networks for the actor and the critic
* using the actor network to run one agent for arbitrarily many episodes and save the observations made
* using the saved observations to train both actor and critic based on the estimated return
 
Even with our simple network architecure we were able to observe a considerable learning effect, finally leading to our agent mastering this simple environment. Although the training result was not stable enough (after several succesful episodes the agent started to get worse again) we decided to not optimize our setup on the CartPole environment, but instead switching to an environment with continous action space and optimizing our learning there. Which leads us to phase 2.

**Phase 2** (implementing A2C for the continuous action space in the LunarLanderContinuous environment):

* changing to the LunarLander gym environment
* deviding the current jupyter notebook into seperate python files(main.py, coordinator.py, agent.py, actor.py and critic.py)
 * the agent now contains the
     * creation of the environment,
     * running an episode and saving the observations
     * computing the gradients for both networks and returning them to the coordinator
 * the coordinator
     * creates the agent
     * tells the agent to run an episode based on the current actor
     * and uses the returned gradients to update the networks   
* modifying the network architecture of the actor to match the new action space: it now has to return two pairs of mean and variance values, each pair describing one normal distribution from which we sample the action for the main and the side engine
* at this point we decided to implement parallel computing of episodes with multiple agents to speed up the learning (because up to this point we were not able to see any useful learning):
 * we looked at different parallelization packages and after some testing we decided to go with [Ray][Ray]
 * Ray allowed us to run multiple agents on our CPUs/GPUs and with this significantly boosting our learning
 
 
 
link tags:

[LLC]: https://gym.openai.com/envs/LunarLanderContinuous-v2/
[CP]: https://gym.openai.com/envs/CartPole-v1/
[Ray]: https://ray.io/

