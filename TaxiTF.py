from DeepQ import *
import gym
import numpy as np

num_episodes = 1000
topology = [500, 12, 12, 6]

deepQ = deepQ_network(topology)
env = gym.make('Taxi-v2')

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
    return np.identity(500)[num:num+1][0]

for i in range(num_episodes):
    total_reward = run_env(True)
    print("Episode: " + str(i) + ", Total Reward: " + str(total_reward))

total_reward = run_env()
print("Total Reward: " + str(total_reward))
