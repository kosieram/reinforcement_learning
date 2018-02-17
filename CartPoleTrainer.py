import gym
import numpy as np

import clr
clr.AddReference("Galapagos")

from System import *
from Galapagos.API import *

env = gym.make("CartPole-v0")

def FitnessFunction(creature):
    nn = creature.GetChromosome[INeuralChromosome]("nn")

    state = env.reset()
    done = False

    total_reward = 0
    while not done:
        #env.render()

        output = nn.Evaluate(state)
        action = np.argmax([output[0], output[1]])

        state, reward, done, info = env.step(action)
        
        total_reward = total_reward + reward

    return total_reward

session = Session.Instance

metadata = session.LoadMetadata("CartPoleMetadata", Func[ICreature, Double](FitnessFunction))
population = session.CreatePopulation(metadata)

population.EnableLogging("CartPoleData")

population.Evolve()

optimalCreature = population.OptimalCreature
nn = optimalCreature.GetChromosome[INeuralChromosome]("nn")

network = nn.ToNeuralNetwork()
annFile = network.Save("CartPoleNetwork")
annFile.WriteToDisk()
