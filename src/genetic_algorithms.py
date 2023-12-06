import torch
import copy
import random
import numpy as np
from torch.utils.data import DataLoader

from src.dataset import WeatherData
from src.weather_lstm import WeatherLSTM

class GeneticAlgorithms():
  """
  Implementation of Genetic Algorithms
  """

  def __init__(self, pop_size, train_data, val_data, device):
    """
    Instantiate Genetic Algorithms with an initial population

    Parameters
    ----------
    pop_size : the population size
    train_data : the training data Pandas DataFrame
    val_data : the validation data Pandas DataFrame
    device : either 'cpu' or 'gpu'
    """
    self.population = self.initial_pop_generator(pop_size)
    self.device = device
    self.train_data = train_data
    self.val_data = val_data
    self.filter_cols = ["outTemp", "barometer", "dewpoint", "outHumidity", "windSpeed10"]


  def initial_pop_generator(self, pop_size):
    """
    Instantiate the initial population randomly within a set of predefined
    ranges.

    Parameters
    ----------
    pop_size : the population size
    """
    population = []
    for i in range(pop_size):
      individual = [
        np.random.randint(1, 128),        # Sequence length(step back)
        np.random.randint(1, 32),         # LSTM hidden size
        np.random.randint(1, 5),          # LSTM number of layers
        np.random.uniform(0.0001, 0.1),   # Optimizer learning rate
        np.sort(                          # Extra parameters
          np.random.choice(["outHumidity", "dewpoint", "barometer", "windSpeed10"], 
            np.random.randint(0,4),
            replace=False
          )
        ).tolist()
      ]

      population.append(individual)

    return population


  def train_model(self, hidden_size, num_layers, learning_rate, extra_params,
                  train_loader, val_loader, num_epochs, method="best", seed=0):
    """
    Run a training loop of LSTM 

    Parameters
    ----------
    pop_size : the population size
    hidden_size : number of hidden neurons in each LSTM layer
    num_layers : number of stacked LSTM layers
    learning_rate : optimizer learning rate
    extra_params : list of additional parameters beyond the outdoor temperature 
      included as features to the input
    train_loader : instance of the PyTorch DataLoader class used to load the 
      training dataset
    val_loader : instance of the PyTorch DataLoader class used to load the 
      validation dataset
    num_epochs : number of epochs to train
    method : either 'best' or 'final'. When 'best' is selected the model with
      the best fitness across all the generations is returned, while when
      'final' is selected only the last trained model is returned
    seed : seed for reproducibility
    """
  
    # Model, loss function, and optimizer
    torch.manual_seed(seed)
    model = WeatherLSTM(input_size=len(extra_params) + 1, 
                        hidden_size=hidden_size, 
                        num_layers=num_layers).to(self.device)

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Lists to store training and validation losses
    best_val_loss = None
    best_model = None

    # Training loop
    for epoch in range(num_epochs):
      
        model.train()

        # Iterate over batches
        for x,y in train_loader:
            
            optimizer.zero_grad()
            outputs = model(x)

            # Compute the loss
            loss = criterion(outputs, y)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            val_loss = 0.0
            num_samples = 0    
            for x,y in val_loader:
                val_outputs = model(x)
                num_samples += val_outputs.size(0)
                val_loss += torch.nn.functional.l1_loss(val_outputs, y, reduction='sum').item()

        average_val_loss = val_loss / (num_samples)
        # print(f'Validation - Epoch [{epoch+1}/{num_epochs}], Loss: {average_val_loss:.4f}')

        # Found a better model, save it
        if method=="best" and (best_val_loss == None or average_val_loss < best_val_loss): 
          best_val_loss = average_val_loss
          best_model = model

    if method=="final":
      best_val_loss = average_val_loss
      best_model = model

    return best_model, best_val_loss


  def fitness_function(self, candidate, num_epochs):
    """
    The fitness function of Genetic Algorithms is the validation loss of the
    model which should be minimized. 

    Parameters
    ----------
    candidate : a list containing all the parameters of the candidate solution
    num_epochs : number of epochs to train the model
    """
    
    # Prepare the dataset and the associated DataLoader using the given
    # candidate sequence length
    seq_len = candidate[0]
    train_dataset = WeatherData(self.train_data, seq_len=seq_len, filter_cols=["outTemp"] + candidate[4], device=self.device)
    val_dataset = WeatherData(self.val_data, seq_len=seq_len, filter_cols=["outTemp"] + candidate[4], device=self.device)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=256)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=256)

    # Train the model and return its loss as the fitness
    _, loss = self.train_model(
                          hidden_size=candidate[1],
                          num_layers=candidate[2],
                          learning_rate=candidate[3],
                          extra_params=candidate[4],
                          train_loader=train_loader,
                          val_loader=val_loader,
                          num_epochs=num_epochs,
                          method="final"
                        )
    return loss

  def get_model(self, candidate, num_epochs):
    """
    Returns the trained model associated with a certain candiate

    Parameters
    ----------
    candidate : a list containing all the parameters of the candidate solution
    num_epochs : number of epochs to train the model
    """
    
    # Prepare the dataset and the associated DataLoader using the given
    # candidate sequence length
    seq_len = candidate[0]
    train_dataset = WeatherData(self.train_data, seq_len=seq_len, filter_cols=["outTemp"] + candidate[4], device=self.device)
    val_dataset = WeatherData(self.val_data, seq_len=seq_len, filter_cols=["outTemp"] + candidate[4], device=self.device)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=256)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=256)

    # Train the model and return it
    model, _ = self.train_model(
                          hidden_size=candidate[1],
                          num_layers=candidate[2],
                          learning_rate=candidate[3],
                          extra_params=candidate[4],
                          train_loader=train_loader,
                          val_loader=val_loader,
                          num_epochs=num_epochs,
                          method="final"
                        )
    return model


  def tournament_selection(self, candidates_fitness, num_tournaments, K=3, 
                            maximize=False):
    """
    Creates 'num_tournaments' tournaments of 'K' individuals randomly chosen
    from the fitness values for the current population and returns the winner of
    each tournament based on whether it is a maximization problem or not.

    Parameters
    ----------
    candidates_fitness : a list containing the fitness value evaluated on each
      individual in the current population
    num_tournaments : number of tournaments to create
    K : the size of each tournament. The smaller, the more explorative Genetic
      Algorithms behaves 
    maximize : specify whether the problem is a maximization problem or not
    """

    pop_size = len(candidates_fitness)
    selected_individuals = []

    # Create N tournaments each made of K candidates taken randomly from the
    # population
    for _ in range(num_tournaments):
      tournament_candidates = np.random.choice(range(pop_size), K, replace=False)

      # Find the winner of the tournament based on fitness
      if maximize:
        tournament_winner = max(tournament_candidates, key=lambda x: candidates_fitness[x])
      else:
        tournament_winner = min(tournament_candidates, key=lambda x: candidates_fitness[x])

      selected_individuals.append(tournament_winner)

    return selected_individuals

  def crossover(self, individual1, individual2):
    """
    Custom crossover operator based on uniform crossover selecting elements from
    the first and second individuals with equal probability.

    Parameters
    ----------
    individual1 : the first individual
    individual1 : the second individual
    """
    final_individual = []
    for i in range(len(individual1)):
      if np.random.rand() > 0.5:  # Equal probability
        final_individual.append(individual1[i])
      else:
        final_individual.append(individual2[i])

    return final_individual

  def ar_crossover(self, individual1, individual2):
    """
    Custom crossover operator based on arithmetic crossover selecting for each
    gene in both individuals the average between them.

    Parameters
    ----------
    individual1 : the first individual
    individual1 : the second individual
    """
    final_individual = []
    for i in range(len(individual1)):
      if i <= 2:
        final_individual.append(int((individual1[i] + individual2[i]) / 2))
      elif i==3:        
        final_individual.append((individual1[i] + individual2[i]) / 2)
      else:
        if np.random.rand() > 0.5:
          final_individual.append(individual1[i])
        else:
          final_individual.append(individual2[i])

    return final_individual

  def mutation(self, individual, mutation_std_dev):
    """
    Custom mutation operator that introduces an integer/float perturbation
    following a normal distribution to the integer/float genes, respectively,
    with a magnitude proportional to a predefined standard deviation provided by
    the 'mutation_std_dev' list. For the last element in the genotype, the set
    of extra parameters, one element is probabilistically removed and added
    based on a certain probability.

    Parameters
    ----------
    individual : the individual to mutate 
    mutation_std_dev : a list of std deviations defining the magnitude of the
      mutations.
    """
    mutated_individual = individual.copy()
    for i in range(len(mutated_individual)):

      # Integer gene : sequence length, LSTM hidden size, LSTM number of layers
      if isinstance(mutated_individual[i], int):
        mutated_individual[i] = round(mutated_individual[i] + np.random.normal(loc=0, scale=mutation_std_dev[i]))
        if mutated_individual[i] < 1: # lower bound
          mutated_individual[i] = 1

      # Floating point gene : optimizer learning rate
      elif isinstance(mutated_individual[i], list):
        rand = np.random.normal(loc=0, scale=mutation_std_dev[i])
        if rand > 1.0:  # delete a random element
          if len(mutated_individual[i]) > 0:
            del mutated_individual[i][np.random.randint(0, len(mutated_individual[i]))]
        elif rand < -1.0: # Add a new random extra variable
          rand_extra_var = self.filter_cols[np.random.randint(1,len(self.filter_cols))]
          if not rand_extra_var in mutated_individual[i]:
            mutated_individual[i].append(rand_extra_var)
        else: # No mutation
          mutated_individual[i] = mutated_individual[i] 

      # Extra parameters list
      else:
        mutated_individual[i] = mutated_individual[i] + np.random.normal(loc=0, scale=mutation_std_dev[i])
        if mutated_individual[i] < 0.0001:  # lower bound
          mutated_individual[i] = 0.0001
    
    return mutated_individual


  def elitism(self, population, fitness, num_elites, maximize=False):
    """
    Picks the 'num_elites' best individuals from the population based on their
    fitness.

    Parameters
    ----------
    population : the population of individuals
    fitness : the fitness value evaluated on each individual in the population
    num_elites : the number of elite individuals to return
    maximize : specify whether the problem is a maximization problem or not
    """

    combined_data = list(zip(population, fitness))

    # Sort the combined data based on fitness
    sorted_data = sorted(combined_data, key=lambda x: x[1], reverse=maximize)

    # Extract the first num_elites individuals from the sorted data
    selected_individuals = [individual for individual, _ in sorted_data[:num_elites]]

    return selected_individuals


  def run(self, num_generations, num_elites, crossover_prob, mutation_prob, 
          mutation_std_dev, maximize=False, num_epochs=15, seed=0):
    """
    Runs genetic algorithm for the predefined number of generations
    'num_generations'.

    Parameters
    ----------
    num_generations : the number of generations to run Genetic Algorithms. This
      is the stopping criterion.
    num_elites : the number of elite individuals in the population
    crossover_prob : the crossover rate, indicating the probability to run
      crossover. This should be a number between [0,1]
    mutation_prob : the mutation rate, indicating the probability to run
      crossover. This should be a number between [0,1]
    mutation_std_dev : a list of std deviations defining the magnitude of the
      mutations.
    maximize : specify whether the problem is a maximization problem or not
    num_epochs : number of epochs to train the model
    seed : seed for reproducibility
    """

    # Track of the population at each generation
    final_pop = None
    final_fit = None
    fitness_evolution = {
      "best": [],
      "avg": [],
      "worst": []
    }

    random.seed(seed)
    np.random.seed(seed)
    
    # Run until the stopping criterion is met
    for gen_i in range(num_generations):
      fitness = []
      print('Generation %d' % gen_i)
      print(self.population)

      # Evaluate the fitness of each individual in the current population
      for candidate in self.population:
        loss = self.fitness_function(candidate, num_epochs=num_epochs)
        fitness.append(loss)

      # Track the avg, best and worst metrics of fitness
      fitness_evolution["avg"].append(np.mean(fitness))
      if maximize:
        fitness_evolution["best"].append(np.max(fitness))
        fitness_evolution["worst"].append(np.min(fitness))
      else:
        fitness_evolution["best"].append(np.min(fitness))
        fitness_evolution["worst"].append(np.max(fitness))

      print("Best %f" % np.min(fitness))

      if gen_i == num_generations-1:
        final_pop = self.population
        final_fit = fitness

      elite_individuals = self.elitism(copy.deepcopy(self.population), fitness, num_elites)

      # Parent selection based on tournament selection
      parents_idx = self.tournament_selection(fitness, K=3, num_tournaments=len(self.population) - num_elites, maximize=maximize)
      parents = [self.population[i] for i in parents_idx]

      # Apply crossover with probability 'crossover_prob'
      for i in range(len(parents)):
        if np.random.rand() < crossover_prob:
          j = np.random.randint(0, len(parents))  # second random individual from the population for crossover
          parents[i] = self.crossover(parents[i], parents[j])
          
      # Apply mutation with probability 'mutation_prob'
      for i in range(len(parents)):
        if np.random.rand() < mutation_prob:
          parents[i] = self.mutation(parents[i], mutation_std_dev)

      # Survivor selection based on elitism
      self.population = elite_individuals + parents
      random.shuffle(self.population)

    
    return final_pop, final_fit, fitness_evolution