import gym
import numpy as np

import clr
clr.AddReference("Galapagos")

from System import *
from Galapagos.API import *

env = gym.make('FrozenLake-v0')

session = Session.Instance
network = session.LoadNeuralNetwork("FrozenLakeNetwork")

state = env.reset()
done = False

while not done:
    env.render()

    output = network.Evaluate(np.identity(16)[state:state+1][0])
    action = np.argmax([output[0], output[1], output[2], output[3]])

    state, reward, done, info = env.step(action)
print(str(state))
