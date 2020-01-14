import array
import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_int", random.randint, 1, 100)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, 2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evalOneMax(individual):
    from runner import get_score
    print(individual)
    obj = type('', (object,), {"ew": individual[0], "ns": individual[1]})()
    return -get_score(obj),


def customCrossover(ind1, ind2):

    int1 = min(ind1[0], ind2[0])
    int2 = max(ind1[0], ind2[0])
    int3 = min(ind1[1], ind2[1])
    int4 = max(ind1[1], ind2[1])

    ind1[0] = random.randint(int1, int2)
    ind2[0] = random.randint(int1, int2)
    ind1[1] = random.randint(int3, int4)
    ind2[1] = random.randint(int3, int4)

    return ind1, ind2


toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", customCrossover)
toolbox.register("mutate", tools.mutUniformInt, low=1, up=100, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():

    pop = toolbox.population(n=30)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=20,
                                   stats=stats, halloffame=hof, verbose=True)

    return pop, log, hof


if __name__ == "__main__":
    main()