---
output:
  html_document: default
  pdf_document: default
---
## Project report

![](report_screenshots/arcade_logo.png){ width=380px }

### General information

**For the course**:

Implementing Artificial Neural Networks (ANNs) with Tensorflow (winter term 2019/20)

**Topic**:

A2C for continuous action spaces applied on the [LunarLanderContinuous][LLC] environment from OpenAI Gym  

**Participants**:

Jonas Otten  
Alexander Prochnow  
Paul Jänsch

### Outline

1. Introduction/Motivation
2. Theoretical Background
3. Project development log
4. The model and the experiment  
 4.1 Original approach with MLP policy network  
 4.2 An alternative actor: Recurrent policy network
5. How to run training and testing
6. Results  
 6.1 MLP  
 6.2 GRU  
7. Discussion
8. References

### 1. Introduction/Motivation

As a final project of this course, one possible task was to identify an interesting research paper in the field of ANNs and reproduce the content of the paper with the knowledge gained during the course. In our case, we first decided on what we wanted to implement and then looked for suitable papers on which we can base our work.  
Inspired by the [lecture about Deep Reinforcement Learning][LeonLect] (RL) held by Leon Schmid we wanted to take the final project as an opportunity to gather some hands-on experience in this fascinating area. So Reinforcement Learning it is. But where to start? We were looking for something which offers a nice tradeoff between accessibility, challenge and chance of success. The [Gym environments][Gym] provided by OpenAI seemed to do just that. Most of them are set up and ready to run within a minute and with them, there is no need to worry about how to access observations or perform actions. One can simply focus on implementing the wanted algorithm.  
Speaking of which, from all the classic DRL techniques we knew so far, the Synchronous Advantage Actor-Critic algorithm (short A2C) seemed most appropriate for the extent of the project. Challenging but doable in a few weeks. This left us with two remaining questions to answer before we could start our project.  
First, which environment exactly should our learning algorithm try to master using A2C? Since there are better solutions than A2C for environments with discrete action spaces, Leon recommended us to go with the [LunarLanderContinuous][LLC] environment.  
And second, which A2C related papers provide us with the necessary theoretical background and also practical inspiration on how to tackle the implementation? An answer to this question we would like to postpone to later sections. Especially the next section about our theoretical background will show on which knowledge our project is built and from where it emerges. Later sections about the practical implementation also contain references to important resources.

### 2. Theoretical Background

In RL an agent is interacting with an environment by observing a state $s_t$ of a state space $S$ and taking an action $a_t$ of an action space $A$ at each discrete timestep $t$. Furthermore, the agent receives a reward $r_t$ at particular timesteps after executing an action. The agents take actions according to a policy $\pi$. In the LunarLanderContinuous environment, the agent receives a reward after each action taken.   

We assume that the environment is modeled by a Markov decision process (MDP), which consists of a state transition function $\mathcal{P}$ giving the probability of transitioning from state $s_t$ to state $s_{t+1}$ after taking action $a_t$ and a reward function $\mathcal{R}$ determining the reward received by taking action $a_t$ in state $s_t$. The *Markov property* is an important element of a MDP, that is the state transition only depends on the current state and action and not on the preceding ones.  
In RL the goal is to maximize the cumulative discounted return at each timestep $t$:

$$G_t = \sum_t^{\infty}{\gamma^{t} r_t}$$

with $\gamma \in (0,1]$ at each timestep $t$. There are two estimates of the return, either the state value function $V^{\pi}(s_t)$ giving the estimated return at state $s_t$ following policy $\pi$ or the state-action value function $Q(s_t, a_t)$ giving the estimated return at state $s_t$ when taking action $a_t$ and following policy $\pi$ afterwards. In classical RL this problem is approached by algorithms which consider each possible state and action in order to find an optimal solution for the policy $\pi$. In continuous state and/or action spaces this approach is computationally too hard. 

In order to overcome this problem function approximation has been used to find a good solution for policy $\pi$, which maximizes the return $G_t$. Common function approximators are deep neural networks (DNNs), which gain raising success in RL as a way to find a good policy $\pi$ in large state and action spaces.  

A big problem in the usage of DNNs for RL is the difficulty of computing the gradient in methods, which estimate the policy $\pi_{\theta}$ with parameters $\theta$ directly. The reward function, which depends on the policy $\pi_{\theta}$, being maximized is defined by:

$$J(\theta) = \sum_{s \in S} d^{\pi}(s) V^{\pi} = \sum_{s \in S} d^{\pi}(s) \sum_{a \in A}{\pi_{\theta}(a|s) Q^{\pi}(s,a))}$$

$d^{\pi}(s)$ is the stationary distribution, that gives the probability of ending up in state $s$ when starting from state $s_0$ and following policy $\pi_{\theta}$. To compute the gradient $\nabla_{\theta}J(\theta)$ it is necessary to compute the gradient of the stationary distribution which depends on the policy and the transition function $d^{\pi}(s) = \lim_{t \to \infty}{\mathcal{P}(s|s_0, \pi_{\theta})}$. Since the environment is unknown this is not possible.   
A reformulation of the gradient of the reward function called the policy gradient theorem (proof: [Sutton & Barto, 2017][PFP]) avoids the calculation of the derivative of the stationary distribution:

$$
\begin{aligned}
\nabla_{\theta} J(\theta) &= \nabla_{\theta} \sum_{s \in S} d^{\pi}(s) \sum_{a \in A}{Q^{\pi}(s,a)) \pi_{\theta}(a|s)} \\ 
&\propto \sum_{s \in S} d^{\pi}(s) \sum_{a \in A}{Q^{\pi}(s,a)) \nabla_{\theta} \pi_{\theta}(a|s)} \\
&= \mathbb{E}_{\pi}[Q^{\pi}(s,a) \nabla_{\theta} \ln{\pi_{\theta}(a|s)}]
\end{aligned}
$$

This formula holds under the assumptions that the state distribution $s \sim d_{\pi_{\theta}}$ and the action distribution $a \sim \pi_{\theta}$ follow the policy $\pi_{\theta}$ (on-policy learning). The action-state value function acts as an incentive for the direction of the policy update and can be replaced by various terms, e.g. the advantage function $A_t$.

An algorithm that makes use of the policy gradient theorem is the actor-critic method. The critic updates the parameters $w$ of a value function and the actor updates the parameters $\theta$ of a policy according to the incentive of the critic. An extension of it is the synchronous advantage actor-critic method (A2C). Here multiple actors are running in parallel. A coordinator waits until each agent is finished with acting in an environment in a specified number of discrete timesteps (synchronous). The received rewards are used to compute the cumulative discounted return $G_t$ for each agent at each timestep. Now we can get an estimate of the advantage $A_t$, that is used as an incentive for the update of the policy: $A^w_t = G_t - V^w_t$. The gradients get accumulated w.r.t. the parameters $w$ of the value function and $\theta$ of the policy:
$$
\begin{aligned}
d\theta &= d\theta + A_w\nabla_\theta \ln{\pi_{\theta}} \\
dw &= dw + \nabla_w(G - V_w)^2
\end{aligned}
$$
These gradients are used to update the parameters of the value function and the policy. After that, all actors start with the same parameters. This algorithm is a variation of the original asynchronous actor-critic method ([A3C][A3C]), where each actor and critic updates the global parameters independently, which leads to actors and critics with different parameters.

Sources used:  
 * the book *Reinforcment Learning: An Introduction* [^1]
 * [the A3C paper][A3C][^2]
 * [Lilian Weng's blogpost about Policy Gradient Algorithms][Lil'Log][^3]
 * [A2C Code provided by OpenAI][A2Code]

### 3. Project development log

Here we describe how we approached the given problem, name the steps we have taken and expound the motivation for the decisions we made in the process of this project. (Readers only interested in the final result with explanations to the important code segments can skip this part and can continue with the paragraph "The model and the experiment")

Instead of directly heading into the complex case of an environment with continuous action space, we decided to first start with a simpler version of A2C. Namely, A2C for a discrete action space and without parallelization. For this, we took the [CartPole][CP] gym environment. Mastering this environment was the objective of phase 1, which also can be seen as a pre-phase to phase 2 (the main phase)

**Phase 1:**

* getting the gym environment to run
* setting up two simple networks for the actor and the critic
* using the actor network to run one agent for arbitrarily many episodes and save the observations made
* using the saved observations to train both actor and critic based on the estimated return
 
Even with our simple network architecture, we were able to observe a considerable learning effect, finally leading to our agent mastering this simple environment. Although the training result was not stable enough (after several successful episodes the agent started to get worse again) we decided to not optimize our setup on the CartPole environment, but instead switching to an environment with continuous action space and optimizing our learning there. Which leads us to phase 2.

**Phase 2:**

* changing to the [LunarLanderContinuous][LLC] gym environment
* dividing the current jupyter notebook into separate python files(main.py, coordinator.py, agent.py, actor.py and critic.py)
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
 * we looked at different parallelization packages and after some testing, we decided to go with [Ray][Ray]
 * Ray allowed us to run multiple agents on our CPUs/GPUs in parallel and with this significantly boosting our learning  
 
With the speed-up provided by the parallelization and further fixes of minor but sometimes critical issues, we were finally able to observe our agents learning useful behaviour in the LunarLander environment up to the point where the Lander actually stopped crashing down on the moon every single time and made its first successful landings. That's one small step for the RL research, one giant leap for our team.  
But we were not quite satisfied with the result yet. The learning process was still very slow and so we decided to add one more ingredient: Long short-term memory or LSTM for short. Adding LSTM to the actor network is said to greatly improve its performance. Further, it might enable our agents to solve other environments, like the [BipedalWalker][BiWalk], which require some kind of longer-lasting memory.  
We advanced into the last phase of our project, which mainly deals with improvements like the implementation of LSTM but also with cleaning, restructuring and polishing the code to achieve its final form.
 
**Phase 3:**

* LSTM implementation:
 * adding pre-build LSTM-Layers by Keras to the actor network
 * expanding the parameter list of the actor's constructor such that one can choose whether the network should use the newly added LSTM layers or the previously used Dense layers
 * the model does not train because states do not get reset in the forward pass during one update, training collapses after reaching a particular threshold at the cumulative return
 * exchanging Keras LSTM-Layers by custom GRU Cells and implement state resetting through masks passed with the input
 * the model trains
 * set up checkpointing for custom layers
* have code infer parameters from environment
* adding an ArgumentParser to the main.py to allow for different settings to be used when calling the main.py (test/training run, type of actor network, number of agents used, type of environment)
* cleaning the code:
 * removing old unused code parts
 * adding necessary comments to the code

 
### 4. The model and the experiment

This section makes up the main part of our report. Here we will highlight and explain the important parts of our project's implementation. We are trying to present the code in the most semantic logical and intuitive order to facilitate comprehension. The code itself is already structured into several classes and we will always indicate which class we are currently talking about. 

### 4.1 Original approach with MLP policy network
We are starting with the coordinator class because, as its name suggests, it organizes the use of every other class and also the whole procedure of the learning process. From there we will go step by step and jump into the other classes as they are coming up.  

![*Figure 1: main.py*](report_screenshots/main.py_coord_init.png)

The instantiation of the coordinator happens in the main.py **(Figure 1)** and the execution of its `__init()__` method initializes everything needed for successful learning. The most crucial point in this part is probably the instantiation of the two Neural Networks which build the core of the A2C method, namely the Actor and the Critic. 

![*Figure 2: coordinator.py init*](report_screenshots/coordinator.py_network_init.png)

As one can see here, network-related things like the loss function and the optimizers are also created at this point. But let's take the chance to go into both classes and look at the architectures of the networks **(Figure 3 & 4)**.

The Critic:

![*Figure 3: critic.py*](report_screenshots/critic.py.png)

The Actor:

![*Figure 4: actor.py*](report_screenshots/actor.py.png)

This is just to gain a quick overview of the networks for now, as we will explain our choice of e.g. activation functions as they become more apparent.  
Back in the `__init()__` method of the coordinator, there is one more important step to talk about. The creation of the agents which will run on the environment in a  parallel manner. 

![*Figure 5: coordinator.py agent instantiations*](report_screenshots/coordinator.py_agent_inst.png)

The instantiation of the agents exhibits an anomaly: the keyword `.remote`. This is necessary because the agent class is declared as a Ray remote class, which has the following implications when instantiated:  

* Instantiation must be done with `Agent.remote()` instead of `Agent()` as seen in the screenshot
* A worker process is started on a single thread of the CPU
* An Agent object is instantiated on that worker
* Methods of the Agent class called on multiple Agents are executed on their respective worker and can therefore execute in parallel, but must be called with `agent_instance.function.remote()`
* Returns of a remote function call now return the task ID, the actual results can be obtained later when needed by calling `ray.get(task_ID)`

After instantiation we assign the first agent to be the "chief" **(Figure 5)**. His environment will be rendered during training, while the environments of the other agents will run in the background. This adds a fun way to watch the performance of our AI, other than graphs and numbers (not that those are not fun, too).  

Besides the Ray specific specialties, the agent class still has a normal `__init()__` method on which we want to have a short glance now:

![*Figure 6: agent.py init*](report_screenshots/agent.py_init.png)

Noteworthy here is the creation of the OpenAI Gym environment in which the agent will act **(Figure 6, Line 11)** and the instantiation of the agent's memory **(Figure 6, Line 22)**. The memory is represented as an object of our Memory class. As expected an object of this class is responsible for storing the observations an agent makes temporally. This includes states visited, actions taken, rewards received, information whether a terminal state is reached and, not being an observation in particular, a return estimate. We will have a look at important methods of the Memory class when we are dealing with the agents executing actions and making observations.


The rest of the coordinator's `__init()__` handles the preparation of the tensorboard in order to be able to inspect the training progress.  
Now that our coordinator is fully operational we can start the training by calling its `train()` method in the main.py **(Figure 1, Line 35)**.   
This method is the heart of the coordinator and will be assisted by quite a lot of helper methods and also some other classes we did not talked about in detail yet. We will go through all of them and explain their use in the order they are needed in the `train()` method.  

![*Figure 7: coordinator.py train*](report_screenshots/coordinator.py_collect_obs.png)

First, we advance the environments a number of timesteps equal to our hyperparameter `num_steps` by calling `step_parallel(t)` accordingly **(Figure 7, Line 73-74)**. Then, in later parts of the `train()` method, we use the collected observations to update the networks. This way we update the network parameters only every `num_steps` (e.g. 32) timesteps. Before we can get into how we update the networks though, we must first have our agents act in the environment and return observations to us. This is the purpose of the `step_parallel(t)` method. It advances all environments from timestep t to t+1 by observing the current state and then computing an action to perform **(Figure 8)**.

![*Figure 8: coordinator.py step_parallel*](report_screenshots/coordinator.py_step_parallel.png)

Observing the current states of the environments of multiple agents can be done in parallel by calling the `agent.observe()` function on all agents **(Figure 8, Line 132)**. Being a remote function, our list comprehension will return a list of task IDs and not the actual states, therefore we must call `ray.get()` to obtain them. Taking a look at the `observe()` function **(Figure 9)** we notice that if we are at the start of a new update, it will reset the agent's memory since we only want to take observations made in the current update cycle into account for the current network update **(Figure 9, Line 26-27)**. We will elaborate on the memory class in the coming section. For now, all we want is the current state, which is stored in the `self.state` attribute of the agent. If the previous episode was finished the environment will be reset and the attribute will instead contain the initial state of the new episode **(Figure 9, Line 30-31)**.

![*Figure 9: agent.py observe*](report_screenshots/agent.py_observe.png)

Returning the current state of every agent to the coordinator, we are now ready to compute our next action for each agent. As described previously: In Lunar Lander, our agent's action consists of two values, one controlling the main engine, the other the side engines. The values can take on virtually any real number within [-1, 1]. We sample these values from two normal distributions per agent **(Figure 8, Line 133-135)**, each with parameters mu and sigma, denoting the location and scale of the distribution. These parameters (one mu and sigma for the main engine distribution and one mu and sigma for the side engines distribution per agent) are the output of our Actor neural network. To compute them, we call the `get_action_distribution()` function **(Figure 10)**, which passes the current states of all environments to the actor network **(Figure 4)**. It returns the mentioned mus and sigmas, which we use to create normal distribution objects **(Figure 10, Line 145)** that will now be sampled from **(Figure 8, Line 135)**.

![*Figure 10: coordinator.py get_action_distribution*](report_screenshots/coordinator.py_get_action_dist.png)

At this point the reasons for our architectural choices for the actor network become apparent: The tanh of the mu output layer **(Figure 4, Line 27)** keeps the center of our normal distributions, i.e. the average of our sampled values within [-1, 1], which is useful since this is exactly the action space. Similarly, for the sigma output layer **(Figure 4, Line 28)**, a softplus activation ensures that the mathematical restrictions of the standard deviation are upheld, namely $\sigma \ge 0$.  

Lastly, to complete our step in the environment, we execute the computed actions **(Figure 8, Line 138)**. A deep dive into the agent's execute function is required before we return the agent's memories to the coordinator. That is because we have to form the agent's memories first. Let us take a look at how this is done.

![*Figure 11: agent.py execute*](report_screenshots/agent.py_execute.png)

The agent performs the action given on the environment and stores the resulting state, reward and done flag returned by the environment, then updates the internal state `self.state` and the finished flag **(Figure 11, Line 50-53)**.  
His observations are stored by the memory object instantiated from our Memory class (memory.py). It is initialized in the agents `__init__` as seen before **(Figure 6, Line 22)** and possesses numpy arrays to store states, actions, rewards, estimated returns and terminal booleans denoting whether a terminal state is reached. The agent's memory starts of empty **(Figure 12, Line 10-15)**. Observations can be stored in the arrays via the index representing the timesteps **(Figure 12, Line 17-22)**.   
   
   
![*Figure 12: memory.py store*](report_screenshots/memory.py_init_store.png)
   
Now that the memories of the agents are filled with the exciting experiences of one timestep, they are eager to return them to the coordinator. But we have taught them well, so that they will only return them to the coordinator when the required `num_steps` is reached **(Figure 11, Line 56-59)**. Until then they repeat the observe and execute routine and only afterwards collectively return a list containing every agent's memory object to the coordinator.
Now it is time for the coordinator to utilize these memories to make the agents better.

![*Figure 13: coordinator.py train*](report_screenshots/coordinator.py_train.png)

We do this by first computing the discounted cumulative return **(Figure 13 Line, 76)**. As described in the theoretical background, our goal is to maximize the cumulative discounted return at each timestep. We had defined it as $G_t = \sum_t^{\infty}{\gamma^{t} r_t}$. For each memory object, we store $G_t$ in the attribute `self.estimated_return`. This is calculated by iterating over the reversed list of rewards **(Figure 14, Line 54)** and summing up all rewards, but discounting future rewards more heavily **(Figure 14, Line 57)**, e.g. $G_1 = R_{t=1} + \gamma \cdot R_{t=2} + \gamma^2 \cdot R_{t=3} + ...$. Our function does this in an unintuitive manner, but it becomes apparent if one would go through this with an example: Looking at a reward list with only 3 entries at the end of an episode: Our `cumulative_return` gets initialized with 0. Our `estimated_return[3]` for timestep 3 is $R_3$ (`self.rewards[3]`). Our `cumulative_return` is now $R_3$. For timestep 2 the estimated return is $R_2 + \gamma \cdot R_3$. Now finally for timestep 1 the discounted cumulative return $G_1$ (our `estimated_return[1]`) is $R_1 + \gamma \cdot (R_2 + \gamma \cdot R_3) = R_1 + \gamma \cdot R_2 + \gamma^2 \cdot R_3$ if we multiply the $\gamma$ into the brackets. This is exactly what we wanted and we hope that the functionality of this method is now more clear. It is important to notice that this estimated return only depends on rewards of following states and not of the ones before.  

Back in the coordinator, concatenating all the made observations across all agents can then be done using the sum function **(Figure 13, Line 77)**, as we have adjusted the memory class's `__add__` behavior method, i.e. what happens when adding two memory objects together, namely that their observations are concatenated.

![*Figure 14: memory.py compute_discounted_cum_return*](report_screenshots/memory.py_compute_return.png)

The memory object of the coordinator now contains the collective memory of all agents and their discounted returns. These are needed to compute the actor loss and critic loss, which we want to minimize, so we compute their gradients. This is coordinated by the `_get_mean_gradients()` function **(Figure 13, Line 80, Figure 15)**. Since we have two networks, two gradients are computed: The policy gradients maximize the actor loss, therefore maximizing the estimated return **(Figure 15, Line 161)**. This might be unconventional since the name "loss" suggests that it should be minimized, but as it is the derivative of the reward function, we want to maximize it. The term contains a log probability and it will in fact converge to 0, because the logarithm of probability values (which are between 0 and 1) is always $\leq 0$. In addition, the advantage term ($A_t^w = G_t -V_t^w$) should also converge to 0 since the estimated return $G_t$ is approximated by our critic (state value function) and we want our critic's estimate to be as close as possible to the actual return.

The critic gradients minimize the critic loss, which will minimize the Mean Squared Error for the state value function **(Figure 15, Line 163)**. 

![*Figure 15: coordinator.py get_mean_gradients*](report_screenshots/coordinator.py_get_gradients.png)  

Let's look at how we calculate the two losses. Firstly, we see that the final actor loss **(Figure 16, Line 170)** is adjusted by an entropy term. Adding the entropy term to the actor loss has been found to improve exploration, which minimizes the risk of convergence to an only locally optimal policy ([A3C paper][A3C] page 4). This adds a new hyperparameter, the entropy coefficient (`ENTROPY_COEF`), which balances the amount of exploration. 

![*Figure 16: coordinator.py compute_gradients*](report_screenshots/coordinator.py_compute_gradients.png)  

The unmodified actor loss is returned from our `_actor_loss` method, which first estimates the state value by passing all states to the critic **(Figure 3)**. In the Lunar Lander environment, a state is a vector of 8 values, denoting different aspects within the environment, e.g. the coordinates of the vessel. So our critic takes this state vector as an input and outputs the state value. Applying L2-regularization has improved our critic loss during training **(Figure 3, Line 10)**.  
The state values are now used to get an estimate of the advantage **(Figure 17, Line 183)**.
But our actor loss also consists of a second part. As described in the theoretical background, our actor gradients are updated via $d\theta = d\theta + A_w\nabla_\theta \ln{\pi_{\theta}}$. We have the advantage $A_w$, now we need to compute the log policy probability $\ln{\pi_\theta}$. Here's how:   
Using our previously mentioned `get_action_distribution` function (**Figure 10**), we recompute the normal distributions that we sampled our performed actions in each respective state from **(Figure 17, Line 186)**.  Inputting the recorded action back into the normal distribution's log probability density function returns us the relative log probability $\ln{\pi_\theta}$ of sampling that action **(Figure 17, Line 187)**. 

![*Figure 17: coordinator.py actor_loss*](report_screenshots/coordinator.py_actor_loss.png)

Our actor loss therefore consists of the log probability and the advantage, which is used as a baseline for the log probability here to reduce the variance of the policy gradients ([A3C paper][A3C] page 3). These gradients must still be computed from the actor loss, which we do by using the tensorflow gradient tape **(Figure 16, Line 177)**.  
Having finished computing the policy gradients, we now move on to the critic gradients **(Figure 15, Line 163)**. For those, we calculate the Mean Squared Error between the discounted cumulative returns **(Figure 14)** and the state values, which are again obtained from pushing the observed states through the critic network **(Figure 16, Line 172-175)**. As we did with the actor loss, gradient of the critic loss is now computed with the help of the gradient tape **(Figure 16, Line 177)** and the policy and critic gradients are now returned to the coordinator **(see below: Figure 13, Line 80)**.

![*Figure 13: coordinator.py train*](report_screenshots/coordinator.py_train.png)

Finally, we can let the Adam optimizers travel through the loss spaces roughly oriented towards the direction of greatest descent given by our gradients, i.e. applying the gradients to update our models **(Figure 13, Line 81-82)**.

Our A2C algorithm has now run through one iteration and performed a single coordinated update of our two networks. All agents will now start their next set of steps in the environment with the same network parameters, which is the before mentioned specialty of the A2C algorithm compared to A3C.


### 4.2 An alternative actor: Recurrent policy network
As an alternative neural network for the policy of the actor, we implemented a recurrent one in order to utilize the advantages of the propagation of hidden states. By doing so, we relax the *Markov property* of the Markov decision process we assume because information about previous states gets taken into account to compute the distribution over the actions. We set up the hypothesis that the information about the impact of previously taken actions on the state of the environment positively influences training speed and task performance.

To set up a recurrent network we used *gated recurrent units* (GRU). These recurrent cells can be seen as a variation of the *Long short-term memory* (LSTM) cell. Similarly, a GRU cell utilizes gates, namely an update gate $z_t$ and a reset gate $r_t$ to avoid vanishing gradients and enable the cell to keep information in the hidden state $h_t$ over an arbitrary amount of time. 

$$
\begin{aligned}
z_t = \sigma(w_z x_t + u_z h_{t-1} + b_z) \\
r_t = \sigma(w_r x_t + u_r h_{t-1} + b_r)
\end{aligned}
$$

with $w$ depicting weight matrices for the input $x_t$, $u$ the weight matrices for the previous hidden state $h_{t-1}$, $b$ the bias and $\sigma$ the sigmoid function. But in contrast to the LSTM, it uses only one hidden state $h_t$ instead of an additional cell state. The calculation of the hidden state/output of the cell integrates the previously computed gates.

$$
\begin{aligned}
{h'}_t &= \text{tanh}(w_h x_t + u_h (r_t \odot h_{t-1})) \\
h_t &= z_t \odot h'_t + (1-z_t) \odot h_{t-1}
\end{aligned}
$$

The complete operations happening in one GRU cell are depicted in **Figure 18** and the notations in **Figure 19**. For a more detailed review of the GRU view the work by [Chung et al. [2014]][^4].

![*Figure 18: GRU Cell*](report_screenshots/gru_cell.png)

![*Figure 19: GRU Cell: Notations*](report_screenshots/gru_cell_notation.png)

In the implementation of a recurrent neural network for the policy we faced some serious issues. The main challenge was imposed by the handling of the hidden states during the collection of observations and in the update of the network. It is difficult to reset the hidden state when the next timestep included in one batch belongs to a new episode. That is the reason why we chose to implement a custom GRU Cell with the Keras API: In order to provide a mask for the hidden state, which resets it at terminal timesteps. This mask just contains zeros for each terminal state and ones for every other state (**Figure 20, Line 121 - 125**). 

![*Figure 20: GRU Cell: executing a parallel step on the environemts*](report_screenshots/coordinator.py_step_parallel_gru.png)

The hidden states returned by the actor (**Figure 4**) are saved between steps of the agent on the environment and between updates. During the update, the shape of the batch is three dimensional. The first dimension represents the agents, the second dimension the timesteps and the last the state vector (**Figure 21**). We found no solution to reset the hidden states of terminal timesteps during the forward pass of multiple timesteps at once without implementing a custom GRU. The recurrent policy returns mu and sigma of the action distribution in the same manner as the feed-forward policy.

![*Figure 21: GRU Cell: getting the action distribution*](report_screenshots/coordinator.py_get_action_dist_gru.png)

We provide the mask together with the input for the recurrent cell and change its shape corresponding to the shape of the hidden state of the cell defined by the number of units  (**Figure 22, Line 73 - 77**). After that, we compute the values of the gates and the current hidden state $h_t$. Then the mask gets forwarded together with the hidden state to the next cell. The hidden state for the next timestep in one batch gets forwarded without the mask because it is provided by the input (**Figure 22, Line 79 - 88**).


![*Figure 22: GRU Cell implementation: forward pass*](report_screenshots/gru.py_call.png)

### 5. How to run training and testing 
Having read our implementation, we hope you are now eager to try it out!  
We have included our trained models, which were saved using the `tf.train.Checkpoint` function. These can be tested by calling `python3 main.py --test`, which will automatically load the trained models and render the environment. We have also added further arguments to the command line parser, which can be viewed using `python3 main.py --help`. Most notably the policy network type can be changed here (`--network_type "mlp"` or `--network_type "gru"`). The number of agents for training can be changed as well, e.g. `--num_agents 12`. To train the model, use `--train`.
Note that this will create new checkpoints throughout the training, meaning that when running `--test` afterwards it will load the new checkpoints and not our trained model anymore.

### 6. Results

In this section we will present the results of our experiment. In particular, we will highlight the performance of the actor with different policy networks. The training runs depicted here where performend with the default parameters given in the code.

On the leaderboards for the OpenAI Gym environments, LunarLanderContinuous-v2 is defined as being "solved" when getting an average reward of 200 over 100 consecutive trials. Both our MLP and GRU networks were able to achieve this. To test this we built a function tracking the reward for each trial into our `main.py --test`. After 100 trials, the program will end and output the average reward over those trials. This has been around 240 for our MLP network after having trained for 4684 episodes (summing the number of episodes each agent performed). Our GRU network has achieved a slightly lower average reward of 229, but it has reached this level of performance after having trained for only 2472 episodes.

### 6.1 MLP

Now we will take a closer look at the performance of the MLP policy during a successful training run.

In **Figure 23** the cumulative return of one agent is plotted against the number of steps taken in parallel after each episode. At the beginning of the training run the variability and the number of datapoints is quite high. This higher data point density happens because the agent crashes the Lunar Lander after a small number of timesteps, whereas later on, when performance has improved, the Lunar Lander will more slowly descend to the surface of the moon, resulting in fewer data points per timesteps. With even greater performance, the agent learns to balance the tradeoff between safe landing and fuel consumption, which results in a negative reward. Thus, the agent tries to land safely with as little fuel as possible, resulting again in a quicker descent and therefore increasing the number of datapoints per step again. Between 200 and 300 the mean cumulative return converges, after about 120,000 timesteps it stops increasing notably and at roughly 200,000 steps the policy starts to overfit. The overfitting after 200,000 timesteps causes the average cumulative return across 100 trials to be 209, which is 31 points lower than our model after 120,000 steps. 
In this case taking a model from an earlier checkpoint (i.e. early stopping) has improved performance.


   ![*Figure 23: mlp cumulative return*](report_screenshots/results/cumulative_return_png.png)  
   
The actor loss (**Figure 24, 25**) given by $\mathbb{E}_{\pi}[A_{w}^{\pi} \ln{\pi_{\theta}(a|s)}]$ oscillates around 0. While the log probability $\ln{\pi_{\theta}(a|s)}$ can only take negative values, the advantage  $A_{w}^{\pi} =  G_t - V^w_t$ can also be positive. The mean of the advantage is similarly around zero, due to the critic being incentivized to bring the advantage, i.e. difference between cumulative return and the critic’s estimate for this return close to zero. 
The log probability converges to zero because the actor tries to take actions with maximal probability, i.e. $\ln({\pi_{\theta}(a|s) = 1})=0$.

   ![*Figure 24: mlp policy loss (main engine)*](report_screenshots/results/policy_main_mlp.png)  
     
   ![*Figure 25: mlp policy loss (side engines)*](report_screenshots/results/policy_side_mlp.png)



Similarly, the critic loss (**Figure 26**) converges to zero as per usual when using Mean Squared Error as the loss function. Of interest here is that the return estimate depends on the policy. That is why the variability of the loss is that high since the policy steadily changes so does the value that the critic tries to estimate. So it is only possible for the critic loss to converge near zero when the policy converges and the actions performed in each state are similar.

   ![*Figure 26: mlp critic loss*](report_screenshots/results/critic_loss_mlp.png)
   

### 6.2 GRU

In this section we will quickly highlight the performance of the GRU policy during training.

We were able to solve the environment with the GRU policy and even faster than with the MLP policy. Nevertheless, with our current configuration training is not stable. The performance of the agent decreases quickly after it starts converging and the cumulative return (**Figure 26**) drops far below zero. 
   
   ![*Figure 26: gru cumulative return*](report_screenshots/results/cumulative_return_gru.png)  
   
The reason why the policy collapses might be that it changes fast and in large steps. Since the critic depends on the policy (return) and vice versa (advantage), too large steps might lead to a downward spiral in which the perfomance decreases rapidly because actor (**Figure 27, 28**) and critic (**Figure 29**) are not able to adapt fast and correct. The policy loss even takes NaN as value at the end because it is dropping to values below the computationally possible.
   
   ![*Figure 27: gru policy loss (main  engine)*](report_screenshots/results/policy_main_gru.png)  
   
   ![*Figure 28: gru policy loss (side  engines)*](report_screenshots/results/policy_main_gru.png)  


   ![*Figure 29: gru critic loss*](report_screenshots/results/critic_loss_gru.png)  


### 7. Discussion

This section will cover our discussion of the project as a whole.
Looking back, we are proud to have trained a model using Reinforcement Learning and implemented the A2C algorithm with efficient parallelization. We could test our understanding of a RL-paper and even improve upon the algorithm using our GRU implementation. Working as a team and collaborating with git has been fun and effective.  
For future projects we would put a greater emphasis on storing summary statistics of training runs more meaningfully to have a backlog of how our model evolves over time. In addition, the implementation could still be improved and extended upon, e.g. by including experience replay to increase sample efficiency [^5]. Also integrating trust region policy optimization from the TRPO algorithm might improve training stability by not increasing the policy parameters too much in a single step [^6]. Furthermore, the GRU network could be replaced by an attention model, e.g. a transformer. Attention networks outperform recurrent networks in many fields, since they maintain utilization of past information, by directing their “attention” towards particular subsets of the data. This enables better parallelization because the data/timesteps must not be forwarded sequentially [^7].

### 8. References
[^1]: Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction.
[^2]: Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., ... & Kavukcuoglu, K. (2016, June). Asynchronous methods for deep reinforcement learning. In International conference on machine learning (pp. 1928-1937).
[^3]: Weng, Lilian. "Policy Gradient Algorithms". In lilianweng.github.io/lil-log. "https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html" (2018).
[^4]: Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555.
[^5]: Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015, June). Trust region policy optimization. In International conference on machine learning (pp. 1889-1897).  
[^6]: Wang, Z., Bapst, V., Heess, N., Mnih, V., Munos, R., Kavukcuoglu, K., & de Freitas, N. (2016). Sample efficient actor-critic with experience replay. arXiv preprint arXiv:1611.01224.  
   
[^7]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).


[LLC]: https://gym.openai.com/envs/LunarLanderContinuous-v2/
[CP]: https://gym.openai.com/envs/CartPole-v1/
[Ray]: https://ray.io/
[Lil'Log]: https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html
[A2Code]: https://github.com/openai/baselines/blob/master/baselines/a2c/a2c.py
[A3C]: https://arxiv.org/pdf/1602.01783.pdf
[Gym]: https://gym.openai.com/
[LeonLect]: https://studip.uni-osnabrueck.de/sendfile.php?type=0&file_id=f0d5efee6a2faf80610f2540611efb47&file_name=IANNwTF_L12_Reinforcement_Learning.pdf
[PFP]: http://incompleteideas.net/book/bookdraft2017nov5.pdf 
[BiWalk]: http://gym.openai.com/envs/BipedalWalker-v2/
