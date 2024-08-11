import math
import random
import networkx as nx
import matplotlib.pyplot as plt
from Customer import create_distance_matrix, read_customers_from_txt


class AntColony:
    """
    This class implements the Ant Colony Optimization (ACO) algorithm
    for solving the Vehicle Routing Problem With Time Window(VRPTW).
    """
    def __init__(self, distance_matrix, customer_array, num_ants, decay, alpha, beta, gamma, capacity_i):
        """
        Initializes the AntColony object with the following parameters:

        - distance_matrix: A 2D matrix representing the distances between customers.
        - customer_array: A list of Customer objects representing the customers.
        - num_ants: The number of ants to use in the colony.
        - decay: The pheromone decay rate.
        - alpha: The weight of the pheromone trails.
        - beta: The weight of the heuristic information.
        - gamma: The weight of the time penalty factor.
        - capacity_i: The vehicle capacity.
        """
        self.pheromone_matrix = [[1 / len(distance_matrix) for _ in range(len(distance_matrix))] for _ in range(len(distance_matrix))]
        """
        Initialize the pheromone matrix with initial values.
        """
        self.distance_matrix = distance_matrix
        self.num_ants = num_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.gamma=gamma
        self.customer_array = customer_array
        self.capacity = capacity_i
        self.tau_0 = 1 / (len(self.distance_matrix) * self.nearest_neighbor_solution_length())
        """
        Calculate the initial global pheromone trail intensity (tau_0).
        """
        print(self.nearest_neighbor_solution_length())
        print(self.tau_0)
        
    def nearest_neighbor_solution_length(self):
        """
        Calculates the length of the nearest neighbor solution.

        The nearest neighbor solution is a simple heuristic where
        each customer is visited once, starting from the depot and
        always visiting the nearest unvisited customer next.

        Returns:
            The length of the nearest neighbor solution.
        """
        num_cities = len(self.customer_array)
        visited = [False] * num_cities
        length = 0
        current_city = 0 
        for _ in range(num_cities - 1):
            next_city = self.nearest_neighbor(current_city, visited)
            length += self.distance_matrix[current_city][next_city]
            visited[current_city] = True
            current_city = next_city
        # Return to the starting city
        length += self.distance_matrix[current_city][0]
        return length

    def nearest_neighbor(self, city, visited):
        """
        Finds the nearest unvisited neighbor of a given city.

        Args:
            city: The index of the current city.
            visited: A list of booleans indicating whether a city has been visited.

        Returns:
            The index of the nearest unvisited neighbor.
        """
        min_distance = float('inf')
        nearest_city = None
        for i, distance in enumerate(self.distance_matrix[city]):
            if not visited[i] and distance < min_distance:
                min_distance = distance
                nearest_city = i
        return nearest_city

    def run(self, iterations):
        """
        Runs the Ant Colony Optimization algorithm for a specified number of iterations.

        Args:
            iterations: The number of iterations to run the algorithm.

        Returns:
            The best route found and its corresponding length and number of vehicles used.
        """
        best_path = None # Initialize the best path found so far
        for i in range(iterations):
            for _ in range(self.num_ants):
                # Generate a path for an ant
                path, num_v, distance = self.generate_path()
                current_solution = (path, num_v, distance)   # Create a tuple to represent the solution
                self.local_update_pheromones(path)  # Update pheromones based on the ant's path
            # Update the best path if necessary
            if best_path is None:
                best_path = current_solution
            else:
                best_path = min([current_solution, best_path], key=lambda x: (x[1], x[2]))  # Compare solutions based on number of vehicles and distance
            print("generate: ", i,"|",best_path[1], best_path[2])
            # Improve the best path using optimization techniques
            improved_best_path=self.optimize_routes(best_path[0])
            best_path = improved_best_path, len(improved_best_path), self.calculate_path_length(improved_best_path)
            print("improved: ",  i,"|",best_path[1], best_path[2], end="\n\n")

            # Update global pheromones based on the best path
            self.global_update_pheromones(best_path) 
        print("final ", best_path[1], best_path[2])
        improved_best_path=self.improve_solution(best_path[0])   # Perform final improvement
        return improved_best_path, len(improved_best_path), self.calculate_path_length(improved_best_path)
    
    def generate_path(self):
        """
        Generates a feasible path for an ant.

        This function constructs a path for an ant by iteratively selecting customers to visit.
        It ensures that the path adheres to vehicle capacity, time window constraints, and
        other problem-specific requirements.

        Returns:
            A tuple containing the generated path, the number of vehicles used, and the total path length.
        """
        capacity = self.capacity  # Initialize vehicle capacity
        not_visited = self.customer_array.copy()  # List of unvisited customers
        path = [[]]  # Initialize path as a list of routes
        time = 0  # Initialize time
        num_V = 1  # Initialize number of vehicles
        start = self.customer_array[0]  # Starting depot
        path[0].append(start)
        not_visited.remove(start)
        possible = not_visited.copy()  # Possible customers to visit

        while len(not_visited) > 0:
            path, possible, time, num_V, capacity = self.update_possible(path, not_visited, capacity, path[-1][-1], self.distance_matrix, time, num_V)
            next_node = self.choose_next_node(path[-1][-1], possible, time)  # Choose the next customer to visit
            not_visited.remove(next_node)
            capacity -= next_node.demand  # Update vehicle capacity
            time += self.distance_matrix[path[-1][-1].cust_no][next_node.cust_no]  # Update time
            time = (next_node.ready_time if time < next_node.ready_time else time) + next_node.service_time  # Consider customer ready time and service time
            path[-1].append(next_node)  # Add customer to the current route
        path[-1].append(self.customer_array[0])  # Return to depot
        return path, num_V, self.calculate_path_length(path)

    def choose_next_node(self, current, possible, time):
        """
        Chooses the next node for an ant to visit based on pheromone levels, visibility, and time penalty.

        This function calculates the probability of visiting each possible customer based on the pheromone levels,
        distance to the customer, and the time penalty for visiting the customer. The next node is selected randomly
        based on these probabilities.

        Args:
            current: The current node (customer) in the path.
            possible: A list of possible customers to visit next.
            time: The current time in the path.

        Returns:
            The chosen next node.
        """
        pheromone_values = self.pheromone_matrix[current.cust_no]
        timeP_values = [((1 / (-((time + self.distance_matrix[current.cust_no][j] ) - self.customer_array[j].ready_time)) + 1e-10) if ((time +  self.distance_matrix[current.cust_no][j] ) - self.customer_array[j].ready_time) < 0 else 1) if any(j == customer.cust_no for customer in possible) else 0 for j in range(len(self.distance_matrix))]
        visibility_values = [1 / (self.distance_matrix[current.cust_no][j] + 1e-10) if any(j == customer.cust_no for customer in possible) else 0 for j in range(len(self.distance_matrix))]
        probabilities = [math.pow(pheromone_values[j], self.alpha) * math.pow(visibility_values[j], self.beta)* math.pow(timeP_values[j], self.gamma) for j in range(len(self.distance_matrix))]
        total_probability = sum(probabilities)
        probabilities = [prob / total_probability for prob in probabilities]
        next_node_index = random.choices(range(len(self.distance_matrix)), weights=probabilities)[0]
        next_node = self.customer_array[next_node_index]
        return next_node


    def calculate_path_length(self, path):
        """
        Calculates the total length of a given path.

        Args:
            path: A list of routes, where each route is a list of customers.

        Returns:
            The total distance of the path.
        """
        length = 0
        for route in path:
            for i in range(len(route) - 1):
                length += self.distance_matrix[route[i].cust_no][route[i+1].cust_no]
        return length

    def global_update_pheromones(self, best_path):
        """
        Updates pheromone levels globally based on the best path.

        This function decreases all pheromone levels by a decay factor and then
        increases the pheromone levels of edges in the best path.

        Args:
            best_path: The best path found so far.
        """
        for i in range(len(self.pheromone_matrix)):
            for j in range(len(self.pheromone_matrix[i])):
                self.pheromone_matrix[i][j] *= (1 - self.decay)

        for route in best_path[0]:
            for i in range(len(route) - 1):
                self.pheromone_matrix[route[i].cust_no][route[i+1].cust_no] += (1 / best_path[2])*self.decay

    def local_update_pheromones(self, path):
        """
        Updates pheromone levels locally based on an ant's path.

        This function decreases pheromone levels on all edges visited by the ant and
        increases the pheromone levels of edges in the ant's path.

        Args:
            path: The path followed by an ant.
        """
        for route in path:
            for i in range(len(route) - 1):
                self.pheromone_matrix[route[i].cust_no][route[i+1].cust_no] *= (1 - self.decay)
                self.pheromone_matrix[route[i].cust_no][route[i+1].cust_no] += (self.decay*self.tau_0)

    def update_possible(self, path, not_visited, capacity, current, time_matrix, time, num_V):
        """
        Updates the list of possible customers to visit based on vehicle capacity and time window constraints.

        This function checks if customers can be served by the current vehicle based on capacity and time window constraints.
        If a vehicle is full or time constraints are violated, a new vehicle is created.

        Args:
            path: The current path.
            not_visited: A list of unvisited customers.
            capacity: The remaining capacity of the current vehicle.
            current: The current customer.
            time_matrix: A matrix of travel times between customers.
            time: The current time.
            num_V: The number of vehicles used so far.

        Returns:
            A tuple containing the updated path, updated list of possible customers, updated time, updated number of vehicles, and updated  capacity.
        """
        can_be_served_by_vehicule = False
        updated_possible = not_visited.copy()

        for customer in not_visited:
            travel_time = time_matrix[current.cust_no][customer.cust_no]
            total_time = time + travel_time

            # Check if due date is late
            if total_time > customer.due_date:
                updated_possible.remove(customer)
                continue
            else:
                can_be_served_by_vehicule = True

            # Check if capacity exceeds the limit
            if capacity - customer.demand < 0:
                updated_possible.remove(customer)
                continue

        if len(updated_possible) == 0:
            path[num_V - 1].append(self.customer_array[0])
            capacity = self.capacity
            time = time + self.distance_matrix[current.cust_no][0]
            if not can_be_served_by_vehicule:
                time = 0
                num_V = num_V + 1
                path.append([self.customer_array[0]])
            path, updated_possible, time, num_V, capacity = self.update_possible(path, not_visited, capacity, self.customer_array[0], time_matrix, time, num_V)

        return path, updated_possible, time, num_V, capacity

########### Improvement Mechanisms ##########

    def improve_solution(self, path):
        """
        Improves a given solution using local search heuristics.

        Applies tour improvement heuristics and the 2-opt heuristic to optimize the given path.

        Args:
            path: The initial path to be improved.

        Returns:
            The improved path.
        """
        # Apply route optimization heuristic
        path = self.optimize_routes(path)
        # Apply 2-opt heuristic for local search
        for i in range(len(path)):
            path[i] = self.two_opt(path[i])
        return path

    def two_opt(self, route):
        """
        Applies the 2-opt heuristic to improve a given route.

        The 2-opt heuristic attempts to improve a route by swapping pairs of edges.

        Args:
            route: The initial route.

        Returns:
            The improved route.
        """
        route_dist = self.route_total_distance(route)
        improved_route = route.copy()
        improved = True
        while improved:
            improved = False
            for i in range(1, len(improved_route) - 3):
                for j in range(i + 1, len(improved_route) - 2):
                    # Swap edges (i, i+1) with (j, j+1)
                    new_route = improved_route[:i] + improved_route[i:j+1][::-1] + improved_route[j+1:]
                    new_route_dist = self.route_total_distance(new_route)
                    if (new_route_dist < route_dist) and self.check_route_valid(new_route):
                        improved_route = new_route
                        route_dist = new_route_dist
                        improved = True
        return improved_route

    def check_route_valid(self, route):
        """
        Checks if a given route is valid according to capacity and time window constraints.

        Args:
            route: The route to be checked.

        Returns:
            True if the route is valid, False otherwise.
        """
        time = 0
        capacity = self.capacity
        for i in range(len(route) - 1):
            current_customer = route[i]
            next_customer = route[i + 1]
            # Calculate travel time from current to next customer
            travel_time = self.distance_matrix[current_customer.cust_no][next_customer.cust_no]
            if next_customer.cust_no == 0:
                capacity = self.capacity
            else:
                # Update time considering service time and ready time constraints
                time = max(next_customer.ready_time, time + travel_time) + next_customer.service_time
                capacity -= next_customer.demand
            # Check if time constraint is violated
            if (time > next_customer.due_date) or (capacity < 0):
                return False  # Route is not valid

        return True  # All constraints satisfied, route is valid

    def route_total_distance(self, route):
        """
        Calculates the total distance of a given route.

        Args:
            route: The route to calculate the distance for.

        Returns:
            The total distance of the route.
        """
        distanceK = 0
        for i in range(len(route) - 1):
            distanceK += self.distance_matrix[route[i].cust_no][route[i+1].cust_no]
        return distanceK

    def optimize_routes(self, routes):
        """
        Optimizes the given routes by exchanging customers between routes.

        This function attempts to improve the solution by moving customers from smaller routes to larger ones.

        Args:
            routes: A list of routes.

        Returns:
            The optimized routes.
        """
        # Sort routes by their lengths
        sorted_routes = sorted(routes, key=lambda x: len(x))
        improved = True
        while improved:
            improved = False
            # Iterate over each small route
            for i in range(len(sorted_routes) - 1):
                new_route1 = sorted_routes[i][:]
                # Iterate over each customer in the small route
                for customer in sorted_routes[i][1:-1]:  # Excluding the depot
                    customer_inserted = False
                    # Try to insert the customer into other routes
                    for j in range(i + 1, len(sorted_routes)):
                        new_route2 = sorted_routes[j][:]
                        # Try inserting the customer at different positions in the route
                        for k in range(1, len(new_route2)):
                            if customer.cust_no != 0:
                                new_route2.insert(k, customer)
                                # Check if the resulting route is valid
                                if self.check_route_valid(new_route2):
                                    # Update the route with the inserted customer
                                    new_route1.remove(customer)
                                    sorted_routes[i] = new_route1
                                    sorted_routes[j] = new_route2
                                    improved = True
                                    customer_inserted = True
                                    break  # Move to the next customer
                                else:
                                    # Revert the insertion if the route is not valid
                                    new_route2.remove(customer)
                            else:
                                customer_inserted = True
                                break
                        if customer_inserted:
                            break  # Move to the next customer
        return sorted_routes



if __name__ == "__main__":
    txt_file_path = './Benchmark/R101.txt'
    customer_array, capacity = read_customers_from_txt(txt_file_path)
    distance_matrix = create_distance_matrix(customer_array)
    num_ants = 10
    decay = 0.1
    alpha = 1.5
    beta = 3
    gamma =2
    iterations = 400
    print(capacity)

    colony = AntColony(distance_matrix, customer_array, num_ants, decay, alpha, beta, gamma, capacity)
    best_path = colony.run(iterations)
    print(best_path[1], best_path[2])

    for pp in best_path[0]:
        for p in pp:
            print(p.cust_no, "-->", end=" ")
        print("\n \n")

# Visualization using networkx
    G = nx.Graph()
    for j in range(len(best_path[0])):
        for i in range(len(best_path[0][j]) - 1):
            G.add_edge(best_path[0][j][i].cust_no, best_path[0][j][i + 1].cust_no)

    pos = {customer.cust_no: (customer.xcoord, customer.ycoord) for customer in customer_array}
    nx.draw(G, pos, with_labels=True, font_weight='bold')
    plt.show()
