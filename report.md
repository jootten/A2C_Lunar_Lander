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
2. Background
3. Project development log
4. The model and the experiment
5. Visualization and results

### Introduction/Motivation

As a final project of this course one possible task was to identify an interesting research paper in the field of ANNs and reproduce the content of the paper with the knowledge gained during the course. In our case we first decided on what we wanted to implement and then looked for suitable papers on which we can base our work.  
Inspired by the [lecture about Deep Reinforcement Learning][LeonLect] (DLR) held by Leon Schmid we wanted to take the final project as an opportunity to gather some hands-on experience in this fascinating area. So Reinforcement Learning it is. But where to start? We were looking for something which offers a nice tradeoff between accessibility, challenge and chance of success. The [Gym environments][Gym] provided by OpenAI seemed to do just that. Most of them are set up and ready to run within a minute and with them there is no need to worry about how to access observations or perform actions. One can simply focus on implementing the wanted algorithm.  
Speaking of which, from all the classic DRL techniques we knew so far, the Synchronous Advantage Actor-Critic algorithm (short A2C) seemed most appropriate for the extent of the project. Challenging but doable in a few weeks. This left us with two remaining questions to answer before we could start our project.  
First, which environment exactly should our learning algorithm try to master using A2C? Since there are better solutions than A2C for environments with discrete action spaces, Leon recommended us to go with the [LunarLanderContinuous][LLC] environment.  
And second, which A2C related papers provide us with the neccessary theoretical background and also practical inspiration on how to tackle the implementation? The answer to this question we want to give in the next section about background knowledge.

### Background

In RL an agent is interacting with an environment by observing a state $s_t$ of a state space $S$ and taking an action $a_t$ of a action space $A$ at each discrete timestep $t$. Furthermore the agent receives a reward $r_t$ at particular timesteps after executing an action. The agents takes the actions accoring to a policy $\pi$. In the LunarLanderContinuous environment the agent receives a reward after each action taken.   
We assume that the environment is modelled by a Markov decision process (MDP), which consists of a state transition function $\mathcal{P}$ giving the probability of transitioning from state $s_t$ to state $s_{t+1}$ after taking action $a_t$ and a reward function $\mathcal{R}$ determining the reward received by taking action $a_t$ in state $s_t$. The *Markov property* is an important element of a MDP, that is the state transition only dependes on the current state and action and not on the precending ones.  
In RL the goal is to maximize the discounted return 

$$G_t = \sum_t^{\infty}{\gamma^{t} r_t}$$

with $\gamma \in (0,1]$ at each timestep $t$. There are two estimates of the return, either the state value function $V^{\pi}(s_t)$ giving the estimated return at state $s_t$ following policy $\pi$ or the state action value function $Q(s_t, a_t)$ giving the estimated return at state $s_t$ when taking action $a_t$ and following policy $\pi$ afterwards. In classical RL this problem is approached by algorithms which consider each possible state and action in order to find an optimal solution for the policy $\pi$. In continuous state and/or action spaces this approch is computionally too hard.   
In order to overcome this problem function approximation has been used to find a good solution for policy $\pi$, that maximizes the return $G_t$. Common function approximators are deep neural networks (DNNs), which gain raising success in RL as a way to find a good policy $\pi$ in large state and action spaces.  
A large problem in the usage of DNNs for RL is the difficulty of computing the gradient in methods which estimate the policy $\pi_{\theta}$ with parameters $\theta$ directly. The reward function, which depends on the policy $\pi_{\theta}$, being maximized is defined by:

$$J(\theta) = \sum_{s \in S} d^{\pi}(s) V^{\pi} = \sum_{s \in S} d^{\pi}(s) \sum_{a \in A}{\pi_{\theta}(a|s) Q^{\pi}(s,a))}$$

$d^{\pi}(s)$ is the stationary distribution, that gives the probability of ending up in state $s$ when starting from state $s_0$ and following policy $\pi_{\theta}$. To compute the gradient $\nabla_{\theta}J(\theta)$ it is necessary to compute the gradient of the stationary distribution which depends on the policy and the transition function $d^{\pi}(s) = \lim_{t \to \infty}{\mathcal{P}(s|s_0, \pi_{\theta})}$, since the environment is unknown this is not possible.   
A reformultation of the derivative of the reward function called the policy gradient theorem (proof: [Sutton & Barto, 2017][PFP]) avoids the calculation of the derivative of the stationary distribution:

$$
\begin{aligned}
\nabla_{\theta} J(\theta) &= \nabla_{\theta} \sum_{s \in S} d^{\pi}(s) \sum_{a \in A}{Q^{\pi}(s,a)) \pi_{\theta}(a|s)} \\ 
&\propto \sum_{s \in S} d^{\pi}(s) \sum_{a \in A}{Q^{\pi}(s,a)) \nabla_{\theta} \pi_{\theta}(a|s)} \\
&= \mathbb{E}[Q^{\pi}(s,a) \nabla_{\theta} \ln{\pi_{\theta}(a|s)}]
\end{aligned}
$$

actor-critic

a3c/a2c

Sources used:
* [the A3C paper][A3C]
* [Lilian Weng's blogpost about Policy Gradient Algorithms][Lil'Log]
* [A2C Code provided by OpenAI][A2Code]

### Project development log

Here we desribe how we approached the given problem, name the steps we have taken and lay out the motivation for the decisions we made in the process of this project. (Readers only interested in the final result with explanations to the important code segments can skip this part and can continue with the paragraph "The model and the experiment")

Instead of directly heading into the complex case of an environment with continuous action space, we decided to first starting with a simpler version of A2C. Namely, A2C for a discrete action space and without parallelization. For this we took the [CartPole][CP] gym environment. Mastering this environment was the objective of phase 1, which also can be seen as a prephase to phase 2 (the main phase)

**Phase 1:**

* getting the gym environment to run
* setting up two simple networks for the actor and the critic
* using the actor network to run one agent for arbitrarily many episodes and save the observations made
* using the saved observations to train both actor and critic based on the estimated return
 
Even with our simple network architecure we were able to observe a considerable learning effect, finally leading to our agent mastering this simple environment. Although the training result was not stable enough (after several succesful episodes the agent started to get worse again) we decided to not optimize our setup on the CartPole environment, but instead switching to an environment with continous action space and optimizing our learning there. Which leads us to phase 2.

**Phase 2:**

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
[Gym]: https://gym.openai.com/
[LeonLect]: https://studip.uni-osnabrueck.de/sendfile.php?type=0&file_id=f0d5efee6a2faf80610f2540611efb47&file_name=IANNwTF_L12_Reinforcement_Learning.pdf
[PFP]: http://incompleteideas.net/book/bookdraft2017nov5.pdf 
