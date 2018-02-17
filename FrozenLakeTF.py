from DeepQ import *
import gym
import numpy as np
import csv

num_episodes = 1000
topology = [16, 16, 4]

deepQ = deepQ_network(topology)
env = gym.make('FrozenLake-v0')

def run_env(train=False):
    percepts = env.reset()
    done = False
    total_reward = 0

    while not done:
        if not train:
            env.render()
        
        action = deepQ.get_action(one_hot(percepts), train)
        next_percepts, reward, done, info = env.step(action)

        if train:
            deepQ.record(one_hot(percepts), action, reward, one_hot(next_percepts), done)

        total_reward = total_reward + reward
        percepts = next_percepts

    if train:
        deepQ.replay()

    return total_reward

def one_hot(num):
    return np.identity(16)[num:num+1][0]

file = open('FrozenLake.csv', 'w')
writer = csv.writer(file)

for i in range(num_episodes):
    total_reward = run_env(True)
    writer.writerow([str(i), str(total_reward)])
    print("Episode: " + str(i) + ", Total Reward: " + str(total_reward))

file.close()
total_reward = run_env()
print("Total Reward: " + str(total_reward))
