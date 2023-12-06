import torch
import random
import operator
import math
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import gp
from deap import tools
from torch.utils.data import DataLoader

from src.dataset import WeatherData

class GeneticProgramming():
  """
  Implementation of Genetic Programming
  """

  def __init__(self, seq_len, pop_size, tournament_size, halloffame_size, 
                X_train, Y_train, X_val, Y_val, device, seed=2):
    """
    Instantiate Genetic Programming by defining the function and terminal sets

    Parameters
    ----------
    seq_len : the lookback size, corresponding to the sequence length
    pop_size : population size
    tournament_size : size of the tournament for selection (parameter K)
    halloffame_size : size of hall of fame
    X_train : the training data X
    Y_train : the training data Y, forecast of the next hour
    X_val : the validation data X
    Y_val : the validation data Y, forecast of the next hour
    device : either 'cpu' or 'gpu'
    """
    self.device = device
    self.X_val = X_val
    self.Y_val = Y_val
    self.seq_len = seq_len

    # Initialize the primitive set, function set
    self.pset = gp.PrimitiveSet("MAIN", seq_len)

    self.pset.addPrimitive(operator.add, 2)
    self.pset.addPrimitive(self.protectedDiv, 2)
    self.pset.addPrimitive(operator.sub, 2)
    self.pset.addPrimitive(operator.mul, 2)
    self.pset.addPrimitive(math.sin, 1)
    self.pset.addPrimitive(math.cos, 1)

    # Add one ephemeral constant 
    try:
        self.pset.addEphemeralConstant("rand101", lambda: random.uniform(0,1))
    except:
        print("EphemeralConstant is already defined, if you changed it restart the kernel")
    try:
        del creator.FitnessMinSR
        del creator.IndividualSR
    except:
        pass
    creator.create("FitnessMinSR", base.Fitness, weights=(-1.0,))
    creator.create("IndividualSR", gp.PrimitiveTree, fitness=creator.FitnessMinSR)

    # Define the initialization methods, in this case generate the tree using
    # the Half-and-Half method
    self.toolbox = base.Toolbox()
    self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=2, max_=3)  
    self.toolbox.register("individual", tools.initIterate, creator.IndividualSR, self.toolbox.expr)
    self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
    self.toolbox.register("compile", gp.compile, pset=self.pset)

    # Fitness function and selection base on Tournaments
    self.toolbox.register("evaluate", self.fitness_function, X_train=X_train, Y_train=Y_train)
    self.toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    # Define crossover and mutation
    self.toolbox.register("mate", gp.cxOnePoint)
    self.toolbox.register("mutate", self.multiple_mutation)

    self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=12))
    self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=12))

    # Seed for reproducibility
    random.seed(seed)

    # Population initialization
    self.population = self.toolbox.population(n=pop_size)
    self.halloffame = tools.HallOfFame(halloffame_size)

    # Statistics
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    self.mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    self.mstats.register("avg", np.mean)
    self.mstats.register("min", np.min)
    self.mstats.register("max", np.max)

  def fitness_function(self, individual, X_train, Y_train):
    """
    Evaluate a given candidate tree(individual) using the training set X_train

    Parameters
    ----------
    individual : the candidate tree
    X_train : the training data
    Y_train : the training data temperature of the next hour
    """
    # Transform the tree expression in a callable function
    gpFunction = self.toolbox.compile(expr=individual)

    # Evaluate the mean absolute error between the predicted value for the
    # temperature and the real one
    mae = 0.0
    for i in range(X_train.size(0)):  
        mae += np.abs(gpFunction(*X_train[i,:,0].flatten().tolist()) - Y_train[i].item())
    
    return mae / (X_train.size(0)),

  def multiple_mutation(self, individual):
    """
    Custom mutation operator that randomly select a strategy with equal
    probability 

    Parameters
    ----------
    individual : the candidate tree to mutate
    """
    roll = random.random()
    if roll <= 0.3333:
      return gp.mutEphemeral(individual, mode="one")
    elif roll <= 0.6666:
      return gp.mutEphemeral(individual, mode="one")
    else:
      return gp.mutInsert(individual, pset=self.pset)


  def protectedDiv(self, left, right):
    """
    Custom division operator to avoid zero division

    Parameters
    ----------
    left : dividend
    right : divisor
    """
    try:
        return left / right
    except ZeroDivisionError:
        return 1


  def run(self, num_generations, crossover_prob, mutation_prob, stats=None):
    """
    Run genetic algorithms for a given number of generations while recording
    statistics. This is an adapted version of the code taken from the official 
    DEAP implementation here: 
    https://github.com/DEAP/deap/blob/master/deap/algorithms.py

    Parameters
    ----------
    num_generations : the number of generations to run Genetic Algorithms. This
      is the stopping criterion.
    crossover_prob : the crossover rate, indicating the probability to run
      crossover. This should be a number between [0,1]
    mutation_prob : the mutation rate, indicating the probability to run
      crossover. This should be a number between [0,1]
    """

    val_mae_evolution = []
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (self.mstats.fields if self.mstats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in self.population if not ind.fitness.valid]
    fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    self.halloffame.update(self.population)

    record = self.mstats.compile(self.population) if self.mstats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, num_generations + 1):
        # Select the next generation individuals
        offspring = self.toolbox.select(self.population, len(self.population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, self.toolbox, crossover_prob, mutation_prob)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        self.halloffame.update(offspring)

        # Replace the current population by the offspring
        self.population[:] = offspring

        # Append the current generation statistics to the logbook
        record = self.mstats.compile(self.population) if self.mstats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        print(logbook.stream)

        # Validation loop
        func = gp.compile(self.halloffame[0], self.pset)
        mae_gp = 0

        starting_point = 100
        for j in range(100):
            i = (starting_point - self.seq_len) + j
            pred = func(*self.X_val[i,:,0].flatten().tolist())
            y = self.Y_val[i]

            # Compute the MAE
            mae_gp += abs(y - pred)
        
        mae_gp /= (j+1)
        val_mae_evolution.append(mae_gp)
        print('MAE GP: %f' % (mae_gp)) 

    return self.population, logbook

  def compileCandidate(self, candidate):
    """
    Compile the given tree into a callable function

    Parameters
    ----------
    candidate : the candidate to compile
    """
    func = gp.compile(candidate, self.pset)
    return func