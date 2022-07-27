# INM707: Deep Reinforcement Learning 
Anthony Hoang and Priyanka Velagala 

For full report, see: [Final-Report](https://github.com/PriyankaVelagala/Deep-Reinforcement-Learning-CW/blob/main/INM707-Hoang-Velagala.pdf)

# Task 1: Grid World 
Q-learning is a model-free reinforcement learning algorithm that allows an agent to learn the value of an action in a
given state. Given a problem that satisfies the conditions of a Markov Decision Process (MDP), this algorithm allows the
agent to learn to act in an optimal manner by learning the consequences of actions through reinforcements offered in
the form of rewards. The environment chosen is a 6x6 grid world, where the objective of the agent is to find the optimal path from location
(0,0) to (5,5) as seen in the figure below. The terrain is designed with additional components the agent must learn about to
maximize rewards along its path. 
![](https://github.com/PriyankaVelagala/Deep-Reinforcement-Learning-CW/blob/main/GridWorld-QLearning/grid_world.png)

# Task 2: Using DDQN to solve Lunar-Lander (OpenGym AI env) 
For the advanced task we chose the Lunar Landing V2 environment provided by OpenAI. At the start of the environment
there is a ship that spawns at the top center of the screen and has an initial velocity that is randomly set. The goal of the
game is to safely land the ship on the landing pad marked by two flags. The game ends if the ship either crashes (body
touches the ground), goes off screen, or comes to rest on the ground.

![](https://github.com/PriyankaVelagala/Deep-Reinforcement-Learning-CW/blob/main/LunarLander-DQN/lunar_lander_screen.png)

With each state involving eight different variables, it is not realistic to keep track of a Q matrix because of the curse of
dimensionality. In order to train an agent to learn how to solve this environment, a deep Q network will be used. 

For our base model we will be implementing a standard DQN with experience replay. This lets the DQN store a short
term memory of transitions and samples a batch from it periodically to train the network. The first improvement upon the 
vanilla DQN is by adding a second DQN so that there is a separate policy network and atarget network. This allows the model to 
use a second network as a target instead of the same one. This avoids the scenario where the target is always changing while training.
The second improvement upon the model we implemented is the prioritized experience replay. This adds another level
of complexity to the experience replay that is already used in the vanilla DQN. The network learns the most from
transitions of states where the loss is high and thus causes the biggest updates in the network’s parameters. 

# Task 3: Using Rainbow-DQN to solve Breakout (Atari) 

# Task 4: Using SAC to solve Cart-Pole (OpenGym AI env) 
