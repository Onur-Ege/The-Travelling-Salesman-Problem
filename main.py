import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.optimize import minimize


# Define the TSP Problem
class TSPProblem(Problem):
    def __init__(self, distance_matrix):
        super().__init__(
            n_var=len(distance_matrix),
            n_obj=1,
            xl=0,
            xu=len(distance_matrix) - 1,
            type_var=int,
        )
        self.distance_matrix = distance_matrix

    def _evaluate(self, x, out, *args, **kwargs):
        total_distances = []
        for route in x:
            route = np.atleast_1d(route).astype(int)
            distance = sum(
                self.distance_matrix[route[i], route[(i + 1) % len(route)]]
                for i in range(len(route))
            )
            total_distances.append(distance)
        out["F"] = np.array(total_distances)


# Load the distance matrix from a file
def load_distance_matrix(filename):
    with open(filename, 'r') as f:
        return np.array([list(map(float, line.split())) for line in f])


# Load city coordinates from a file for visualization
def load_city_coordinates(filename):
    with open(filename, 'r') as f:
        return np.array([list(map(float, line.split()[1:])) for line in f])


# Print the final result and plot the route
def print_and_plot_result(problem, algorithms, city_coordinates):
    for name, algorithm in algorithms.items():
        print("")
        print(name)
        print("")

        # Minimize the problem using the algorithm
        result = minimize(problem, algorithm, seed=algorithm.n_gen, verbose=False)

        # Extract the best route and distance
        best_route = result.X
        best_distance = result.F[0].item()  # Ensure it's a scalar value

        print(f"Best Route: {best_route + 1}")  # Convert to 1-based index
        print(f"Total Distance: {best_distance:.2f}")

        # Plot the best route
        plot_route(best_route, best_distance, city_coordinates, name)


# Plotting function
def plot_route(route, distance, city_coordinates, algorithm_name):
    plt.figure()
    route = np.append(route, route[0])  # Close the loop
    x = city_coordinates[route, 0]
    y = city_coordinates[route, 1]

    # Plot the route
    plt.plot(x, y, marker="o", linestyle="-", color="b", label="Route")

    # Annotate each city with its number above the node
    for i, (x_coord, y_coord) in enumerate(zip(x[:-1], y[:-1])):  # Exclude the last point (duplicate of the start)
        city_number = route[i] + 1  # Convert to 1-based index
        plt.text(
            x_coord, y_coord + 0.02,  # Offset the text slightly above the node
            str(city_number),
            color="black",
            fontsize=8,
            ha="center",
            va="bottom"
        )

    # Highlight the starting city
    plt.scatter(x[0], y[0], color="green", label="Start/End City", s=150)  # Starting and ending city

    # Add title and labels
    plt.title(f"{algorithm_name} - Best Route - Distance: {distance:.2f}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # Load the distance matrix and city coordinates
    distance_matrix = load_distance_matrix('intercityDistance.txt')
    city_coordinates = load_city_coordinates('cityData.txt')

    # Define the starting cities (e.g., first 5 cities)
    starting_cities = range(5)  # You can change this to any 5 cities you want to start with

    # Configure the algorithms
    algorithms = {
        "Genetic Algorithm": GA(
            pop_size=150,
            sampling=PermutationRandomSampling(),
            crossover=OrderCrossover(),
            mutation=InversionMutation(),
            eliminate_duplicates=True,
        ),
        "NSGA-II": NSGA2(
            pop_size=150,
            sampling=PermutationRandomSampling(),
            crossover=OrderCrossover(),
            mutation=InversionMutation(),
        )
    }

    # Run both algorithms for each starting city
    for starting_city in starting_cities:
        print(f"Running for starting city: {starting_city + 1}")
        # Modify the problem to fix the starting city
        problem = TSPProblem(distance_matrix)
        problem.X0 = np.array([starting_city])  # Fix the starting city (optional)

        # Print and plot the results for both algorithms
        print_and_plot_result(problem, algorithms, city_coordinates)


if __name__ == "__main__":
    main()


