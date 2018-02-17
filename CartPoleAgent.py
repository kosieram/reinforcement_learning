import gym
import numpy as np

import clr
clr.AddReference("Galapagos")

from System import *
from Galapagos.API import *

env = gym.make("CartPole-v0")

session = Session.Instance
network = session.LoadNeuralNetwork("CartPoleNetwork")

state = env.reset()
done = False

while not done:
    env.render()

    output = network.Evaluate(state)
    action = np.argmax([output[0], output[1]])

    state, reward, done, info = env.step(action)

