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

# Task 2: Using DDQN to solve Lunar-Lander (OpenAI Gym env) 
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
The game chosen for this task was Breakout from the Atari Learning Environment. The 128 byte RAM from the console
was used for training. This observation space stores both the state of the environment as well as the call stack among
other components. The objective of the game is to destroy the wall of bricks on the top of the screen by hitting the ball against them. On
hitting a brick, the ball bounces towards the bottom of the screen and the player must adjust the paddle such that the
ball bounces back towards the bricks or the ball disappears to the bottom of the screen and the player loses a life.

![](https://github.com/PriyankaVelagala/Deep-Reinforcement-Learning-CW/blob/main/Atari-Breakout/breakout_screen.png)

The RL Algorithm implemented from rllib was Rainbow DQN. This algorithm works in discrete action spaces as is the case
for Breakout. It also uses off-policy learning that allows it to learn from a set of transitions in memory. This allows the
algorithm to have good sampling efficiency as the agent can learn from the same state transition multiple times before
discarding from memory. This algorithm uses a neural network to learn the best action to take in a given state by calculating the Q-value of each
action and then uses an action selection policy (e.g. epsilon-greedy) to select an action. To set hyperparameters, ray.tune
was used to perform a grid search over 4 extensions of DQN introduced in Rainbow DQN - Double-DQN, Prioritized Experience Replay, Noisy Nets and Dueling Nets.
To find the optimal combination of hyper-parameters for this environment, the 4 enhancements were evaluated on a vanilla-DQN with a single 128 fully connected layer with ReLU activation. 


# Task 4: Using SAC to solve Cart-Pole (OpenAI Gym env) 
Soft Actor Critic is one of the newer reinforcement learning algorithms and claims to be more efficient in terms of
samples needed to learn than the traditional RL algorithms. The main characteristic of SAC is that while it maximizes
rewards for the environment, it also attempts to maximize the entropy of the policy. The term entropy refers to the
randomness of the action selection of the policy. By maximizing this randomness, the algorithm encourages maximum
exploration of the environment space. It does so by limiting the probabilities assigned to the actions with higher Q values
and keeps weights closer to other actions with similar Q values (V.Kumar, 2019). This policy was trained on the CartPole-v0 from OpenAI Gym. 
Hyperparameters were set based on benchmark configuration settings provided by the rllib for this trainer.

![](https://github.com/PriyankaVelagala/Deep-Reinforcement-Learning-CW/blob/main/CartPole-SAC/cart_pole_screen.png)
