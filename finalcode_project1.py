# import required libraries
import random  # for random operations in genetic algorithm
import matplotlib.pyplot as plt  # for plotting results
import numpy as np  # numerical operations
from scipy.integrate import solve_ivp  # solving differential equations
import os  # handling directory operations for saving results

# genetic algorithm parameters
POPULATION_SIZE = 100  # number of bacteria individuals in each generation
MUTATION_RATES = [0.1, 0.075, 0.05, 0.03, 0.01]  # mutation rates to compare
GENERATIONS = 25  # number of evolutionary generations
SELECTION_METHOD = (
    "tournament"  # or "roulette" selection strategy (not actually used in current code)
)
TOURNAMENT_SIZE = 3  # number of candidates in tournament selection
NUM_EXPERIMENTS = 3  # number of experimental repeats for statistics

# parameter ranges for bacterial traits
LAG_TIME_RANGE = (2, 8.0)  # acceptable lag time range in hours (not directly used)
GROWTH_RATE_RANGE = (0.1, 1.0)  # growth rate range per hour (not directly used)
growth_values = np.linspace(0.4, 5, 100)  # predefined growth rate options
lag_values = np.linspace(0, 10, 100)  # predefined lag time options


# initialize population with random growth rates and lag times
def initialize_population():
    population = []
    for _ in range(POPULATION_SIZE):
        # select correlated growth/lag values from predefined lists
        growth_lag_index = random.randint(0, 99)
        lag_time = lag_values[growth_lag_index]
        growth_rate = growth_values[growth_lag_index]
        population.append({"lag_time": lag_time, "growth_rate": growth_rate})
    return population


# calculate fitness by simulating growth in experimental conditions
def evaluate_fitness(individual, experimental_cycle):
    # initial conditions: equal starting populations and resources
    x = np.array([10**3, 10**3])  # initial bacterial populations (two species)
    s = np.array([2000, 2000])  # initial resource concentrations (two resources)

    # convert individual's parameters to numpy arrays for vector operations
    lag = np.array([individual["lag_time"], individual["lag_time"]])
    mu = np.array([individual["growth_rate"], individual["growth_rate"]])

    # run simulation and sum final populations of both species
    dict_of_pop = ODE_set_up(experimental_cycle, x, s, lag, mu)
    final_population = sum(
        [dict_of_pop[key][-1][-1] for key in dict_of_pop.keys() if "Bac" in key]
    )
    return final_population


# parent selection using tournament or roulette method
def select_parents(population, fitness_scores):
    # check selection method and choose parents accordingly
    if SELECTION_METHOD == "roulette":
        # calculate total fitness for probability distribution
        total_fitness = sum(fitness_scores)
        # create probability weights based on fitness scores
        probabilities = [f / total_fitness for f in fitness_scores]
        # select two parents using weighted random choice
        parents = random.choices(population, weights=probabilities, k=2)
    elif SELECTION_METHOD == "tournament":
        parents = []
        for _ in range(2):
            # randomly select candidates for tournament
            tournament = random.sample(
                list(zip(population, fitness_scores)), TOURNAMENT_SIZE
            )
            # choose winner with highest fitness score
            winner = max(tournament, key=lambda x: x[1])[0]
            parents.append(winner)
    return parents


# create offspring by randomly choosing traits from parents
def crossover(parent1, parent2):
    child = {
        "lag_time": random.choice([parent1["lag_time"], parent2["lag_time"]]),
        "growth_rate": random.choice([parent1["growth_rate"], parent2["growth_rate"]]),
    }
    return child


# apply mutation with probability=mutation_rate using growth-lag tradeoff
def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        # attempt to modify both traits together
        growth, lag = growth_tradeoff(individual)
        individual["lag_time"] = lag
        individual["growth_rate"] = growth
    return individual


# enforce tradeoff between growth rate and lag time
def growth_tradeoff(individual):
    try:
        # randomly modify both traits using same index offset
        change_var = random.uniform(-10, 10)
        current_growth_rate_index = np.where(
            growth_values == individual["growth_rate"]
        )[0]
        growth_change = growth_values[current_growth_rate_index + change_var]
        current_lag_index = np.where(lag_values == individual["lag_time"])[0]
        lag_change = lag_values[current_lag_index + change_var]

        return growth_change, lag_change
    except IndexError:
        # handle cases where index would be out of bounds
        pass
        return individual["growth_rate"], individual["lag_time"]


# execute one generation of genetic algorithm
def evolve_population(population, experimental_cycle, mutation_rate):
    # evaluate all individuals
    fitness_scores = [
        evaluate_fitness(individual, experimental_cycle) for individual in population
    ]
    new_population = []
    # create new population through selection/crossover/mutation
    for _ in range(POPULATION_SIZE):
        parent1, parent2 = select_parents(population, fitness_scores)
        child = crossover(parent1, parent2)
        child = mutate(child, mutation_rate)
        new_population.append(child)
    return new_population, fitness_scores


# differential equation system setup and solver
def ODE_set_up(experimental_cycle, x, s, lag, mu):
    # resource utilization rates matrix
    util = np.array([[0.009, 0.009], [0.009, 0.009]])

    # simulation parameters
    simulation_run = 24  # hours per growth cycle
    time_step_space = 1000  # resolution for ODE solution
    dilution = 50  # dilution factor between cycles
    dict_of_pop = {
        "Bac 1": [],
        "Bac 2": [],
        "R1": [],
        "R2": [],
    }  # storage for population/resource data

    initial = [pop for pop in x]  # initial bacterial populations

    # simulate each resource cycle in experimental design
    for resource in experimental_cycle:
        # prepare initial conditions with current resource concentration
        initial.append(s[resource])
        # solve differential equations
        solver = solve_ivp(
            solve_switch_system,
            t_span=[0, simulation_run],
            y0=initial,
            t_eval=np.linspace(1, simulation_run, time_step_space),
            method="LSODA",
            args=(util[resource], mu[resource], lag[resource]),
        )

        # store bacterial population data
        for state in range(len(solver.y[:2])):
            keys = list(dict_of_pop.keys())
            vals = list(dict_of_pop[keys[state]])
            vals.append(solver.y[state].tolist())
            dict_of_pop[keys[state]] = vals

        # store resource consumption data
        for unused in range(len(s)):
            if unused == resource:  # track active resource
                vals = dict_of_pop[f"R{unused + 1}"]
                vals.append(solver.y[-1].tolist())
                dict_of_pop[f"R{unused + 1}"] = vals
            else:  # unused resources remain zero
                unused_vals = dict_of_pop[f"R{unused + 1}"]
                unused_vals.append([0 for _ in range(len(solver.y[1]))])
                dict_of_pop[f"R{unused + 1}"] = unused_vals

        # prepare next cycle with diluted populations
        initial_updated = []
        for i in range(len(x)):
            keys = list(dict_of_pop.keys())
            final_val = dict_of_pop[keys[i]][len(dict_of_pop[keys[i]]) - 1]
            initial_updated.append(final_val[-1] / dilution)  # apply dilution

        initial = initial_updated

    return dict_of_pop


# differential equations for bacterial growth system
def solve_switch_system(t, initial, util_sim, mu_sim, lag):
    x_sim = initial[:2]  # bacterial populations
    x_sim = np.array(x_sim)
    s_sim = initial[2:]  # resource concentrations

    # growth equations with lag phase modulation
    dx = Logistic(t, lag) * (mu_sim * (s_sim / (s_sim + 50))) * x_sim
    # resource consumption equation
    dr = np.dot((-(util_sim * (s_sim / (s_sim + 50)))), x_sim)

    return [*dx, dr]  # combine results


# logistic function modeling lag phase transition
def Logistic(time, TTE):
    # sigmoid function centered at TTE/2 with steepness 5
    log = 1 / (1 + np.exp(-5 * (time - (TTE / 2))))
    return log


# run complete experiment with multiple mutation rates
def run_experiment(experiment_dir):
    experimental_cycle = [0, 1, 0, 1, 0, 1]  # resource alternation pattern
    experiment_fitness = {}  # store fitness by mutation rate

    # test each mutation rate
    for mutation_rate in MUTATION_RATES:
        mr_dir = os.path.join(experiment_dir, f"mutation_{mutation_rate}")
        os.makedirs(mr_dir, exist_ok=True)  # create output directory

        population = initialize_population()
        fitness_history = []  # track average fitness per generation
        table_data = []  # data for summary table

        # evolve through generations
        for generation in range(GENERATIONS):
            population, fitness_scores = evolve_population(
                population, experimental_cycle, mutation_rate
            )
            avg_fitness = np.mean(fitness_scores)
            best_individual = population[np.argmax(fitness_scores)]

            # record data for current generation
            table_data.append(
                [
                    mutation_rate,
                    generation + 1,
                    avg_fitness,
                    best_individual["lag_time"],
                    best_individual["growth_rate"],
                ]
            )
            fitness_history.append(avg_fitness)

            # print progress
            print(
                f"Mutation Rate {mutation_rate} - Generation {generation + 1}: Avg Fitness = {avg_fitness:.4f}, "
                f"Lag Time = {best_individual['lag_time']:.4f}, Growth Rate = {best_individual['growth_rate']:.4f}"
            )

        # save fitness progression plot
        plt.figure()
        plt.plot(
            range(1, GENERATIONS + 1),
            fitness_history,
            marker="o",
            linestyle="-",
            linewidth=2,
        )
        plt.title(f"Mutation Rate {mutation_rate} Fitness Progression")
        plt.savefig(os.path.join(mr_dir, "fitness_progression.png"))
        plt.close()

        # save results table as image
        fig, ax = plt.subplots(figsize=(8, GENERATIONS * 0.5))
        ax.axis("off")
        ax.table(
            cellText=table_data,
            colLabels=[
                "Mutation",
                "Generation",
                "Avg Fitness",
                "Lag Time",
                "Growth Rate",
            ],
            loc="center",
        )
        plt.savefig(os.path.join(mr_dir, "results_table.png"), bbox_inches="tight")
        plt.close()

        experiment_fitness[mutation_rate] = fitness_history

    # create combined plot for all mutation rates
    plt.figure()
    for mutation_rate, fitness in experiment_fitness.items():
        plt.plot(fitness, label=f"Mutation {mutation_rate}")
    plt.title("Experiment Fitness Progression")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.savefig(os.path.join(experiment_dir, "combined_fitness.png"))
    plt.close()

    return experiment_fitness


# main execution block
if __name__ == "__main__":
    all_data = {mutation_rate: [] for mutation_rate in MUTATION_RATES}

    # run multiple experiments for statistical significance
    for experiment in range(NUM_EXPERIMENTS):
        experiment_dir = f"experiment_{experiment + 1}"
        experiment_data = run_experiment(experiment_dir)
        # aggregate data across experiments
        for mutation_rate, fitness in experiment_data.items():
            all_data[mutation_rate].append(fitness)

    # calculate average fitness across experiments
    avg_fitness = {mutation_rate: [] for mutation_rate in MUTATION_RATES}
    for mutation_rate in MUTATION_RATES:
        for generation in range(GENERATIONS):
            avg = (
                sum(
                    all_data[mutation_rate][experiment][generation]
                    for experiment in range(NUM_EXPERIMENTS)
                )
                / NUM_EXPERIMENTS
            )
            avg_fitness[mutation_rate].append(avg)

    # plot final comparison of all mutation rates
    plt.figure(figsize=(10, 6))
    for mutation_rate in MUTATION_RATES:
        plt.plot(avg_fitness[mutation_rate], label=f"Mutation {mutation_rate}")
    plt.title("Average Fitness Across All Experiments")
    plt.xlabel("Generation")
    plt.ylabel("Average Fitness")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig("average_fitness_across_experiments.png")
    plt.show()
