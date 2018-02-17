import gym
import numpy as np

import clr
clr.AddReference("Galapagos")

from System import *
from Galapagos.API import *

env = gym.make("Taxi-v2")
state = env.reset()

def FitnessFunction(creature):
    nn = creature.GetChromosome[INeuralChromosome]("nn")

    total_reward = 0
    state = env.reset()
    done = False
        
    while not done:
        #env.render()
        output = nn.Evaluate(np.identity(500)[state:state+1][0])
        action = np.argmax([output[0], output[1], output[2], output[3], output[4], output[5]])

        state, reward, done, info = env.step(action)

        total_reward = total_reward + reward

    return total_reward

session = Session.Instance
metadata = session.LoadMetadata("TaxiMetadata", Func[ICreature, Double](FitnessFunction))
population = session.CreatePopulation(metadata)

population.EnableLogging("TaxiData")

population.Evolve()

optimalCreature = population.OptimalCreature
nn = optimalCreature.GetChromosome[INeuralChromosome]("nn")

network = nn.ToNeuralNetwork()
annFile = network.Save("TaxiNetwork")
annFile.WriteToDisk()
