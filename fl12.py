# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 17:38:29 2021

@author: ja2
"""
import time
import gym
import numpy as np 


env = gym.make('FrozenLake8x8-v0')
iterations=22222
gam=1

States = [0 for i in range(env.nS)]
newStates = States.copy()
for k in range(iterations):
    for state in range(env.nS):
      action_values = []      
      for action in range(env.nA):
        StateValue = 0
        for i in range(len(env.P[state][action])):
          prob, next_state, reward, done = env.P[state][action][i]
          state_action_value = prob * (reward + gam*States[next_state])
          StateValue += state_action_value
        action_values.append(StateValue)      
        best_action = np.argmax(np.asarray(action_values))   
        newStates[state] = action_values[best_action] 
    if i > iterations/2: 
      if abs(sum(States) - sum(newStates)) < 0.01:   
        break
    else:
      States = newStates.copy()
      
policy = [0 for i in range(env.nS)]
for state in range(env.nS):
    action_values = []
    for action in range(env.nA):
      action_value = 0
      for i in range(len(env.P[state][action])):
        prob, next_state, r, _ = env.P[state][action][i]
        action_value += prob * (r + gam * States[next_state])
      action_values.append(action_value)
    best_action = np.argmax(np.asarray(action_values))
    policy[state] = best_action


holes = 0
steps_list = []

for episode in range(iterations):
    observation = env.reset()
    steps=0
    while True:
      action = policy[observation]
      observation, reward, done, _ = env.step(action)
      steps+=1
      if done and reward == 1:
        steps_list.append(steps)
        break
      elif done and reward == 0:
        holes += 1
        break



time.sleep(1)
curr_state=0
observation = env.reset()
i=1
while True:
        env.render()
        print(i)
        i+=1
        action = policy[observation]
        observation, reward, done, _ = env.step(action)        
        time.sleep(1)
        if done or reward==1:
            break
env.render()

print('average steps in success {:.0f}  '.format(np.mean(steps_list)))
print('failed {:.2f} % '.format((holes/iterations) * 100))
