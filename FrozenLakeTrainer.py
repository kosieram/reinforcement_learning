import gym
import numpy as np

import clr
clr.AddReference("Galapagos")

from System import *
from Galapagos.API import *

env = gym.make("FrozenLake-v0")

def FitnessFunction(creature):
    nn = creature.GetChromosome[INeuralChromosome]("nn")

    total_reward = 0
    for i in range(0, 5):
        state = env.reset()
        done = False
        
        while not done:
            #env.render()

            output = nn.Evaluate(np.identity(16)[state:state+1][0])
            action = np.argmax([output[0], output[1], output[2], output[3]])

            state, reward, done, info = env.step(action)

            total_reward = total_reward + reward

    return total_reward

session = Session.Instance
metadata = session.LoadMetadata("FrozenLakeMetadata", Func[ICreature, Double](FitnessFunction))
population = session.CreatePopulation(metadata)

population.EnableLogging("FrozenLakeData")

population.Evolve()

optimalCreature = population.OptimalCreature
nn = optimalCreature.GetChromosome[INeuralChromosome]("nn")

network = nn.ToNeuralNetwork()
annFile = network.Save("FrozenLakeNetwork")
annFile.WriteToDisk()
