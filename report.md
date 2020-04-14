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
2. Theoretical Background
3. Project development log
4. The model and the experiment
5. Visualization and results

### 1. Introduction/Motivation

As a final project of this course one possible task was to identify an interesting research paper in the field of ANNs and reproduce the content of the paper with the knowledge gained during the course. In our case we first decided on what we wanted to implement and then looked for suitable papers on which we can base our work.  
Inspired by the [lecture about Deep Reinforcement Learning][LeonLect] (DLR) held by Leon Schmid we wanted to take the final project as an opportunity to gather some hands-on experience in this fascinating area. So Reinforcement Learning it is. But where to start? We were looking for something which offers a nice tradeoff between accessibility, challenge and chance of success. The [Gym environments][Gym] provided by OpenAI seemed to do just that. Most of them are set up and ready to run within a minute and with them there is no need to worry about how to access observations or perform actions. One can simply focus on implementing the wanted algorithm.  
Speaking of which, from all the classic DRL techniques we knew so far, the Synchronous Advantage Actor-Critic algorithm (short A2C) seemed most appropriate for the extent of the project. Challenging but doable in a few weeks. This left us with two remaining questions to answer before we could start our project.  
First, which environment exactly should our learning algorithm try to master using A2C? Since there are better solutions than A2C for environments with discrete action spaces, Leon recommended us to go with the [LunarLanderContinuous][LLC] environment.  
And second, which A2C related papers provide us with the neccessary theoretical background and also practical inspiration on how to tackle the implementation? The answer to this question we want to give in the next section about background knowledge.

### 2. Theoretical Background

In RL an agent is interacting with an environment by observing a state $s_t$ of a state space $S$ and taking an action $a_t$ of an action space $A$ at each discrete timestep $t$. Furthermore the agent receives a reward $r_t$ at particular timesteps after executing an action. The agents takes the actions accoring to a policy $\pi$. In the LunarLanderContinuous environment the agent receives a reward after each action taken.   

We assume that the environment is modelled by a Markov decision process (MDP), which consists of a state transition function $\mathcal{P}$ giving the probability of transitioning from state $s_t$ to state $s_{t+1}$ after taking action $a_t$ and a reward function $\mathcal{R}$ determining the reward received by taking action $a_t$ in state $s_t$. The *Markov property* is an important element of a MDP, that is the state transition only depends on the current state and action and not on the precending ones.  
In RL the goal is to maximize the cumulative discounted return at each timestep $t$:

$$G_t = \sum_t^{\infty}{\gamma^{t} r_t}$$

with $\gamma \in (0,1]$ at each timestep $t$. There are two estimates of the return, either the state value function $V^{\pi}(s_t)$ giving the estimated return at state $s_t$ following policy $\pi$ or the state-action value function $Q(s_t, a_t)$ giving the estimated return at state $s_t$ when taking action $a_t$ and following policy $\pi$ afterwards. In classical RL this problem is approached by algorithms which consider each possible state and action in order to find an optimal solution for the policy $\pi$. In continuous state and/or action spaces this approch is computionally too hard. 

In order to overcome this problem function approximation has been used to find a good solution for policy $\pi$, that maximizes the return $G_t$. Common function approximators are deep neural networks (DNNs), which gain raising success in RL as a way to find a good policy $\pi$ in large state and action spaces.  

A big problem in the usage of DNNs for RL is the difficulty of computing the gradient in methods, which estimate the policy $\pi_{\theta}$ with parameters $\theta$ directly. The reward function, which depends on the policy $\pi_{\theta}$, being maximized is defined by:

$$J(\theta) = \sum_{s \in S} d^{\pi}(s) V^{\pi} = \sum_{s \in S} d^{\pi}(s) \sum_{a \in A}{\pi_{\theta}(a|s) Q^{\pi}(s,a))}$$

$d^{\pi}(s)$ is the stationary distribution, that gives the probability of ending up in state $s$ when starting from state $s_0$ and following policy $\pi_{\theta}$. To compute the gradient $\nabla_{\theta}J(\theta)$ it is necessary to compute the gradient of the stationary distribution which depends on the policy and the transition function $d^{\pi}(s) = \lim_{t \to \infty}{\mathcal{P}(s|s_0, \pi_{\theta})}$, since the environment is unknown this is not possible.   
A reformulation of the gradient of the reward function called the policy gradient theorem (proof: [Sutton & Barto, 2017][PFP]) avoids the calculation of the derivative of the stationary distribution:

$$
\begin{aligned}
\nabla_{\theta} J(\theta) &= \nabla_{\theta} \sum_{s \in S} d^{\pi}(s) \sum_{a \in A}{Q^{\pi}(s,a)) \pi_{\theta}(a|s)} \\ 
&\propto \sum_{s \in S} d^{\pi}(s) \sum_{a \in A}{Q^{\pi}(s,a)) \nabla_{\theta} \pi_{\theta}(a|s)} \\
&= \mathbb{E}_{\pi}[Q^{\pi}(s,a) \nabla_{\theta} \ln{\pi_{\theta}(a|s)}]
\end{aligned}
$$

This formula holds under the assumptions that the state distribution $s \sim d_{\pi_{\theta}}$ and the action distribution $a \sim \pi_{\theta}$ follow the policy $\pi_{\theta}$ (on-policy learning). The action-state value function acts as an incentive for the direction of the policy update and can be replaced by various terms, e.g. the advantage function $A_t$.

An algorithm that makes use of the policy gradient theorem is the actor-critic method. The critic updates the parameters $w$ of a value function and the actor updates the parameters $\theta$ of a policy according to the incentive of the critic. An extension of it is the synchronous advantage actor-critic method (A2C). Here multiple actors are running in parallel. A coordinator waits until each agent is finished with acting in an environment in a specified number of discrete timesteps (synchronous). The received rewards are used to compute the cumulative discounted return $G_t$ for each agent at each timestep. No we can get an estimate of the advantage $A_t$, that is used as incentive for the update of the policy: $A^w_t = G_t - V^w_t$. The gradients get accumulated w.r.t. the parameters $w$ of the value function and $\theta$ of the policy:
$$
\begin{aligned}
d\theta &= d\theta + A_w\nabla_\theta \ln{\pi_{\theta}} \\
dw &= dw + \nabla_w(G - V_w)^2
\end{aligned}
$$
These gradients are used to update the parameters of the value function and the policy. After that all actors start with the same parameters. This algorithm is a variation of the original asynchronous actor-critic method ([A3C][A3C]), where each actor and critic updates the global parameters independently, which leads to actors and critics with different parameters.

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
 
**Phase 3:**

* added LSTM-Layers to Actor network for better performance
* Have code infer parameters from environment


 
### The model and the experiment

This section makes up the main part of our report. Here we will highlight and explain the important parts of our project's implementation. We are trying to present the code in the most semantic logical and intuitiv order to facilitate the comprehension. The code itself is already structured into several classes and we will always indicate which class we are currently talking about.  
We are starting with the coordinator class because, as its name suggests, it organizes the use of every other class and also the whole procedure of the learning process. From there we will go step by step and jump into the other classes as they are coming up.  
The instantiation of the coordinator happens in the main.py **(main.py line 21)** and the execution of its constructor initializes everything needed for successful learning. The most crucial point in this part is probably the instantiation of the two Neural Networks which build the core of the A2C method, namely the Actor and the Critic. **(coordinator.py line 42-49)** As one can see here, network related things like the loss function and the optimizers are also created at this point. But let's take the chance to go into both classes and look at the architectures of the networks.

The Critic:

... **(screenshots of critic.py)** 

The Actor:

... **(screenshots of actor.py)** 

Back in the init() method of the coordinator, there is one more important step to talk about. The creation of the agents which will run on the environment in a  parallel manner. **(coordinator.py line 51-52)** The instantiation of the agents exhibits an anomaly: the keyword ".remote". This is necessary, because the agent class is declared as a Ray remote class, which has the following implications when instantiated:  

* Instantiation must be done with `Agent.remote()` instead of `Agent()` as seen in the screenshot
* A worker process is started on a single thread of the GPU
* A Agent object is instantiated on that worker
* Methods of the Agent class called on multiple Agents can execute in parallel, but must be called with `agent_Instance.function.remote()`
* Returns of a remote function call now return the task ID, the actual results can be obtained later when needed by calling `ray.get(task_ID)`

The rest of the coordinator's init() just deals with the preperation of the thensorboard in order to be able to inspect the training progress.  
Now that our coordinator is fully operational we can start the training by calling its train() method in the main.py **(main.py line 22)**   
This method is the heart of the coordinator and will be assisted by quite a lot helper methods and also some other classes we did not talked about in detail yet. We will go through all of them and explain their use in the order they are needed in the train() method.  

possible structure in train():

* step_parallel() only mlp part  
 * agent's observe()
 * get_action_distribution()
 * agent's execute() 
   * take a look at the agent's init() because of the memory object initialized there and talk about the memory class itself
* compute_discounted_cum_return() of memory class in line 69 of coordinator
* get_mean_gradients() in line 72 of coordinator
 * compute_gradients()
   * actor_loss()
   * entropy()
* apply_gradients to the networks in line 73-74
* the part storing summary statistics
* the part with checkpoints



 
### Visualization and results

...

[LLC]: https://gym.openai.com/envs/LunarLanderContinuous-v2/
[CP]: https://gym.openai.com/envs/CartPole-v1/
[Ray]: https://ray.io/
[Lil'Log]: https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html
[A2Code]: https://github.com/openai/baselines/blob/master/baselines/a2c/a2c.py
[A3C]: https://arxiv.org/pdf/1602.01783.pdf
[Gym]: https://gym.openai.com/
[LeonLect]: https://studip.uni-osnabrueck.de/sendfile.php?type=0&file_id=f0d5efee6a2faf80610f2540611efb47&file_name=IANNwTF_L12_Reinforcement_Learning.pdf
[PFP]: http://incompleteideas.net/book/bookdraft2017nov5.pdf 
