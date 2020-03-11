## Project report

### General informations

**For the course**:

Implementing Artificial Neural Networks (ANNs) with Tensorflow (winter term 2019/20)

**Topic**:

A2C for continuous action spaces applied on the [LunarLanderContinuous][LLC] environment from Gym OpenAI

**Participants**:

Jonas Otten  
Alexander Prochnow  
Paul Jänsch

### Outline

1. Introduction/Motivation
2. Important background knowledge
3. Project development log
4. The model and the experiment
5. Visualization and results

### Introduction/Motivation

As a final project of this course our task was to identify an interesting research paper in the field of ANNs and reproduce the content of the paper with the knowledge gained during this course. Inspired by the lecture about Reinforcement learning held by Leon S. we wanted to take the final project as an opportunity to gather some hands-on experience in this fascinating area.

### Important background knowledge

Sources used:
* [the A3C paper][A3C]
* [Lilian Weng's blogpost about Policy Gradient Algorithms][Lil'Log]
* [A2C Code provided by OpenAI][A2Code]

### Project development log

Here we desribe how we approached the given problem, name the steps we have taken and lay out the motivation for the decisions we made in the process of this project. (Readers only interested in the final result with explanations to the important code segments can skip this part and can continue with the paragraph "The model and the experiment")

Instead of directly heading into the complex case of an environment with continuous action space, we decided to first starting with a simpler version of A2C. Namely, A2C for a discrete action space and without parallelization. For this we took the [CartPole][CP] gym environment. Mastering this environment was the objective of phase 1, which also can be seen as a prephase to phase 2 (the main phase)

**Phase 1**:

* getting the gym environment to run
* setting up two simple networks for the actor and the critic
* using the actor network to run one agent for arbitrarily many episodes and save the observations made
* using the saved observations to train both actor and critic based on the estimated return
 
Even with our simple network architecure we were able to observe a considerable learning effect, finally leading to our agent mastering this simple environment. Although the training result was not stable enough (after several succesful episodes the agent started to get worse again) we decided to not optimize our setup on the CartPole environment, but instead switching to an environment with continous action space and optimizing our learning there. Which leads us to phase 2.

**Phase 2**:

* changing to the [LunarLanderContinuous][LLC] gym environment
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
 
### The model and the experiment

...
 
### Visualization and results

[LLC]: https://gym.openai.com/envs/LunarLanderContinuous-v2/
[CP]: https://gym.openai.com/envs/CartPole-v1/
[Ray]: https://ray.io/
[Lil'Log]: https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html
[A2Code]: https://github.com/openai/baselines/blob/master/baselines/a2c/a2c.py
[A3C]: https://arxiv.org/pdf/1602.01783.pdf
