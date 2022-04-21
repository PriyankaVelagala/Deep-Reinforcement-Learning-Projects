# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 10:24:20 2022

@author: priya
"""

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
import time 
import pandas as pd 
import seaborn as sns
import random
sns.set()

"""
Obstacles 
"""
BIDIRECTIONAL_WALLS = [(4,5), (18,24), (17,23), (30,31)] 
ONE_WAY_WALLS = [(7,1), (6, 12) ]
ONE_WAY_TUNNELS = [(6,30), (7,5)]

"""
Reward Structure
"""
DEFAULT_REWARD = -1 
REWARD = 25 #50
REWARD_LOC = [5, 13, 16, 32]
PENALTY = -100
PENALTY_LOC = [8, 17, 26, 30, 34] 
DST_REWARD = 1000
START = 0 
DST = 35


class grid_world():
    
    def __init__(self, logger):
        self.debug_mode = logger
        self.S = list(range(0,36))
        self.A = list(range(0,36))
        #self.walls =  [(4,5),  (18,24), (17,23), (30,31)] #[(3,4), (3,9), (4,5), (11,17), (17,23), (12,18), (18,24), (30,31), (27,33)]
        self.legal_moves = self.get_legal_moves()
        self.R = self.initialize_rewards()
        self.Q = np.zeros(self.R.shape)
        print("Grid world initialized!")
        
        
        
    def get_legal_moves(self):
        possible_actions = []
        illegal_moves = [] 

        #all adjacent locations for each cell 
        for row in range(0,6):
            for col in range(0, 6):
                current = row * 6 + col
        
                if col > 0:
                    left = current - 1
                    possible_actions.append((current, left))
                if col < 5:
                    right = current + 1
                    possible_actions.append((current, right))
                if row > 0:
                    up = current - 6
                    possible_actions.append((current, up))
                if row < 5:
                    down = current + 6
                    possible_actions.append((current, down))
                    
        #add tunnels 
        for tunnel in ONE_WAY_TUNNELS:
            possible_actions.append((tunnel[0], tunnel[1]))
        
        if self.debug_mode: print("# moves:", len(possible_actions))
        
        #add walls 
        for wall in BIDIRECTIONAL_WALLS:
            illegal_moves.append((wall[0], wall[1]))
            illegal_moves.append((wall[1], wall[0]))
            
        for wall in ONE_WAY_WALLS:
            illegal_moves.append((wall[0], wall[1]))
                    
    
        if self.debug_mode: print("# illegal moves:", len(illegal_moves))   

        for illegal_move in illegal_moves:
            possible_actions.remove(illegal_move)

        if self.debug_mode: 
            print("# legal moves:", len(possible_actions))
            print(possible_actions)
        
        return possible_actions


    def initialize_rewards(self):
        R = np.empty((len(self.A), len(self.S)))
        R[:] = np.nan
        
        for action in self.legal_moves:    
            if action[1] in REWARD_LOC: # Rewards
                R[action[0], action[1]] = REWARD
            elif action[1] in PENALTY_LOC: # Penalties
                R[action[0], action[1]] = PENALTY 
            elif action[1] == DST:  #destination
                R[action[0], action[1]] = DST_REWARD
            else:
                R[action[0], action[1]] = DEFAULT_REWARD
                
        return R
    
    
    def reset_Q(self):
        self.Q = np.zeros(self.R.shape)
        return 1
        
    
    
    """
    Epsilon-Greedy Policy 
    low epsilon - higher chance of exploitation 
    high epsilon - higher chance of exploration 
    """
    def decay_eps_greedy(self, eps, all_actions, best_actions): 
        if np.random.uniform() > eps:
            #exploit
            a = np.random.choice(best_actions)
        else:
            #explore
            a = np.random.choice(all_actions )
            
        return a
    
    
    """
    Softmax Policy 
    """
    def softmax(self, current_state, actions, T):
        if self.debug_mode:
            for action in actions:
                print(f'Q[{current_state},{action}] = {self.Q[current_state,action]}')

        probabilities = np.array([self.Q[current_state, a]/T for a in actions])
        if self.debug_mode: print(f'probabilities = {probabilities}')
    
        softmax_probabilities = np.exp(probabilities)/ np.sum(np.exp(probabilities))
        if self.debug_mode: print(f'softmax_probabilities= {softmax_probabilities}')
    
        # choose an action based on these probabilities
        a = random.choices(actions, weights=softmax_probabilities)
        
        return a[0]
    
    
    
        
        

    """
    Q-learning algo 
    """
    def run_q_learning(self, num_episodes, steps, alpha, gamma, policy, policy_args = {}):
        print(policy)
        time_start = time.time()
        
        #reset Q 
        self.reset_Q()
        #self.Q = np.zeros(self.R.shape)
        
        rewards_per_episode = []
        steps_per_episode = []
        found_flag = 0 
        
        path = []
        final_rewards = [] 
        

            
      
        if policy == 'eps_greedy' or policy == 'eps_greedy_decay' :
            decay_factor = policy_args.pop('decay_rate')   
            eps = [] 
            eps_start = 0.9
            eps.append(eps_start)
            for i in range(1, num_episodes):
                eps.append(eps[i-1]*decay_factor)
        else:
            tau = policy_args.pop('tau') 

    

    
        for i in range(num_episodes):    
            # Initialize State
            s = 0 #np.random.choice(36) #DO NOT RANDOMIZE INITIAL STATE: CANNOT ANALYZE AVG. STEPS-REWARDS/EPISODE 
            
            total_reward = 0
    
            if i%100 == 0 and self.debug_mode:
                print('Running episode {} ....'.format(i))
                    
            for step in range(steps):
                available_actions = np.where(~np.isnan(self.R[s]))[0]
                q_values = [self.Q[s,a] for a in available_actions]
                
                if policy == 'eps_greedy' or policy == 'eps_greedy_decay': 
                    best_actions = available_actions[np.where(q_values == np.max(q_values))[0]]
                    if self.debug_mode: print(f"Currently at {s}. best actions: {best_actions}. Step: {step}")
                
                    #best_actions_q_values = [Q[s,x] for x in best_actions]
                    
                    a = self.decay_eps_greedy(eps[i], available_actions, best_actions)
                    if self.debug_mode : print(f"choose: {a}")

                elif policy == 'softmax':
                    a = self.softmax(s, available_actions, tau) #tau[i])
                    if self.debug_mode: print(f"Currently at {s}. action selected: {a}. Step: {step}")

                
                r = self.R[s,a]
                
                
                if i == (num_episodes-1):
                    path.append(a)
                
                if a in REWARD_LOC or a == DST  : # Reward/Flag found
                    if self.debug_mode : print(f"Found reward. Adding {max((r - step),0) }")
                    total_reward += max((r - step),0)  # Discount reward based on steps
                    if i == (num_episodes-1):
                        final_rewards.append(max((r - step),0))
                else: #penalty or normal step 
                    total_reward += r
                    if i == (num_episodes-1):
                        final_rewards.append(r)      
                
                if self.debug_mode: print(f"new reward total: {total_reward}")
                s_old = s
                s = a 
    
                # Q value updating
                q_updated = self.Q[s_old,a] + alpha * ( r + gamma * np.max(self.Q[s,:]) - self.Q[s_old,a])
                
                if self.Q[s_old,a] != q_updated and self.debug_mode:
                    print(f"Old q value:  { self.Q[s_old,a]}, new Q value: {q_updated}")
                    
                self.Q[s_old,a] = q_updated

                #for softmax, normalize Q before next iteration to avoid overflow 
                #if policy == 'softmax':
                #  self.Q = self.Q/np.max(np.abs(self.Q)) #initially when rewards are negtive ,causes divide by 0, use abs 
    
                if self.S[s] == DST: # Destination Reached
                    rewards_per_episode.append(total_reward)
                    steps_per_episode.append(step)
                    found_flag += 1 
                    break
                elif step == (steps -1): #out of steps
                    rewards_per_episode.append(total_reward)
                    steps_per_episode.append(step)
    
        print(f"time elapsed for {num_episodes} episodes : {time.time()-time_start} s")
        print("Path: ", path)
        print("Rewards: ", final_rewards)
        print(f"Total rewards: {sum(final_rewards)} , Total path length: {len(path)}" )
            
        return rewards_per_episode, steps_per_episode, found_flag, self.Q
    
    
    def plot_graph(self, df, title, x_label, y_label, legend_title = None):
        sns.set(rc = {'figure.figsize':(15,5)})
        sns.set_style("whitegrid", {'axes.grid' : False})

        dashes_val = ['' for x in df.columns]

        f = sns.lineplot(data=df, dashes = dashes_val)

        f.set_xlabel(x_label, fontsize = 15)
        f.set_ylabel(y_label, fontsize = 15)
        f.set_title(title)

        if legend_title: 
          leg = f.axes.get_legend()
          leg.set_title(legend_title)
              
                                  
                        