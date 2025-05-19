import random
import matplotlib.pyplot as plt

# Function to read preferences from file
def read_preferences(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    preferences = [list(map(int, line.split())) for line in lines]

    # Decrease the values by one to match Python indexing
    preferences = [[p - 1 for p in line] for line in preferences]
    return preferences[:30], preferences[30:]

# Function to calculate the fitness of a solution
def calculate_fitness(solution, male_prefs, female_prefs):
    max_fitness = 1740  # The maximum fitness value
    fitness = 0
    # Calculate the fitness value for the solution
    for m, w in enumerate(solution):
        fitness += male_prefs[m].index(w) + female_prefs[w].index(m)
    return (1 - (fitness / max_fitness)) * 100  # Return fitness as a percentage

# Function to calculate the probability of each solution being selected based on fitness
def calculate_prob_fitnesses(fitnesses):
    total_fitness = sum(fitnesses)
    return [f / total_fitness for f in fitnesses]

# Function to initialize the initial population
def initialize_population(pop_size):
    population = []
    for _ in range(pop_size):
        females = random.sample(range(30), 30)
        population.append(females)
    return population

# Function to perform crossover between two parents and ensure no duplicates
def crossover(parent1, parent2):
    for attempt in range(15):  # Try up to 15 times
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + [w for w in parent2[point:] if w not in parent1[:point]]
        child2 = parent2[:point] + [w for w in parent1[point:] if w not in parent2[:point]]

        if len(child1) == len(set(child1)) == 30 and len(child2) == len(set(child2)) == 30:
            return child1, child2

    # If no valid crossover found after 15 attempts, return parents unchanged
    return parent1, parent2

# Function to perform mutation on a solution
def mutate(solution):
    if len(solution) != 30:
        raise ValueError("Solution length must be 30.")
    i, j = random.sample(range(30), 2)
    solution[i], solution[j] = solution[j], solution[i]

# Genetic algorithm function
def genetic_algorithm(male_prefs, female_prefs, pop_size=100, generations=100):
    # Initialize the population
    population = initialize_population(pop_size)
    best_solution = None
    best_fitness = float('-inf')
    worst_solution = None
    worst_fitness = float('inf')
    total_fitness = 0

    best_fitness_list = []
    worst_fitness_list = []
    average_fitness_list = []
    # Run the genetic algorithm for the specified number of generations
    for generation in range(generations):
        fitness_scores = [calculate_fitness(sol, male_prefs, female_prefs) for sol in population]
        population = sorted(population, key=lambda sol: calculate_fitness(sol, male_prefs, female_prefs), reverse=True)

        current_best_fitness = calculate_fitness(population[0], male_prefs, female_prefs)
        current_worst_fitness = calculate_fitness(population[-1], male_prefs, female_prefs)
        current_total_fitness = sum(fitness_scores)

        # Update best solution
        if current_best_fitness > best_fitness:
            best_solution = population[0]
            best_fitness = current_best_fitness

        # Update worst solution
        if current_worst_fitness < worst_fitness:
            worst_solution = population[-1]
            worst_fitness = current_worst_fitness

        # Accumulate total fitness for average calculation
        total_fitness += current_total_fitness

        # Adjust mutation rate
        if generation > 0 and current_best_fitness == best_fitness == worst_fitness:
            mutation_rate = 0.7
        else:
            mutation_rate = 0.1

        # Building the next generation
        next_population = population[:pop_size // 20]  # saving 5% of the best solutions
        prob_fitnesses = calculate_prob_fitnesses(fitness_scores)  # Calculate the probability of each solution being selected

        while len(next_population) < pop_size:
            # Select a parent based on fitness
            parent1 = random.choices(population, weights=prob_fitnesses, k=1)[0]
            parent2 = random.choices(population, weights=prob_fitnesses, k=1)[0]
            if random.random() < 0.9: # 90% chance of crossover
                child1, child2 = crossover(parent1, parent2)
                if random.random() < mutation_rate: # 10% chance of mutation
                    mutate(child1)
                    mutate(child2)
            else:
                # Copy the parents if no crossover
                child1 = parent1[:]
                child2 = parent2[:]
                # Mutate the children
                if random.random() < mutation_rate:
                    mutate(child1)
                    mutate(child2)
            # Add the children to the next population
            next_population.append(child1)
            next_population.append(child2)

        population = next_population
        # Save the fitness values for plotting
        best_fitness_list.append(current_best_fitness)
        worst_fitness_list.append(current_worst_fitness)
        average_fitness_list.append(current_total_fitness / pop_size)
    # Calculate the average fitness
    average_fitness = total_fitness / (pop_size * generations)
    # Return the best and worst solutions and their fitness values
    return best_solution, best_fitness, worst_solution, worst_fitness, average_fitness, best_fitness_list, worst_fitness_list, average_fitness_list

# Function to plot the fitness values over generations
def plot_fitness(best_fitness_list, worst_fitness_list, average_fitness_list, pop_size, generations, avg_satisfaction):
    generations_range = range(len(best_fitness_list))
    plt.figure(figsize=(10, 6))  # Make the plot larger
    plt.plot(generations_range, best_fitness_list, label='Best Fitness')
    plt.plot(generations_range, worst_fitness_list, label='Worst Fitness')
    plt.plot(generations_range, average_fitness_list, label='Average Fitness')
    plt.xlabel('Generations')
    plt.ylabel('Fitness (%)')
    plt.legend()
    plt.title(f'Fitness over Generations\n(pop_size={pop_size}, generations={generations})\nAverage satisfaction for the best Solution: {avg_satisfaction:.2f}')
    plt.tight_layout()
    plt.show()

def main():
    # Read the preferences from the file
    male_prefs, female_prefs = read_preferences('.\GA_input.txt')
    # Run the genetic algorithm
    pop_size = 60
    generations = 300
    best_solution, best_fitness, worst_solution, worst_fitness, average_fitness, best_fitness_list, worst_fitness_list, average_fitness_list = genetic_algorithm(
        male_prefs, female_prefs, pop_size=pop_size, generations=generations)
    # Calculate the average satisfaction for the best solution
    avg_satisfaction_best = (1 - (best_fitness / 100)) * 1740 / 60
    # Print the results
    print("Best Solution:", [(m + 1, w + 1) for m, w in enumerate(best_solution)])
    print("Best Fitness:", best_fitness, "%")
    print("Average satisfaction for the best Solution:", avg_satisfaction_best)
    print("Worst Fitness:", worst_fitness, "%")
    print("Average Fitness:", average_fitness, "%")

    plot_fitness(best_fitness_list, worst_fitness_list, average_fitness_list, pop_size, generations, avg_satisfaction_best)
    # Input("Press Enter to exit...")  # Pause the program to prevent immediate closure
if __name__ == "__main__":
    main()
