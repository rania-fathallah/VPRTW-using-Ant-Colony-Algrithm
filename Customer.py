import csv
import math

# Global variable storing the vehicle's speed
vehicle_speed = 1

class Customer:
    """Represents a customer with its attributes."""
    def __init__(self, cust_no, xcoord, ycoord, demand, ready_time, due_date, service_time):
        """
        Initializes a Customer object.

        Args:
            cust_no: Unique identifier for the customer.
            xcoord: X-coordinate of the customer's location.
            ycoord: Y-coordinate of the customer's location.
            demand: The customer's demand for a product or service.
            ready_time: The time when the customer becomes available for service.
            due_date: The deadline for completing service for the customer.
            service_time: The time required to service the customer.
        """
        self.cust_no = cust_no
        self.xcoord = xcoord
        self.ycoord = ycoord
        self.demand = demand
        self.ready_time = ready_time
        self.due_date = due_date
        self.service_time = service_time

def read_customers_from_csv(file_path):
    """
    Reads customer data from a CSV file.

    Args:
        file_path: Path to the CSV file containing customer information.

    Returns:
        A list of Customer objects representing the data from the CSV file.
    """
    customers = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Extract customer data from the CSV row
            cust_no = int(row['CUST NO.'])
            xcoord = float(row['XCOORD.'])
            ycoord = float(row['YCOORD.'])
            demand = int(row['DEMAND'])
            ready_time = int(row['READY TIME'])
            due_date = int(row['DUE DATE'])
            service_time = int(row['SERVICE TIME'])

            # Create a Customer object and append it to the list
            customer = Customer(cust_no, xcoord, ycoord, demand, ready_time, due_date, service_time)
            customers.append(customer)
    return customers

def read_customers_from_txt(file_path):
    """
    Reads customer data from a TXT file.

    Args:
        file_path: Path to the TXT file containing customer information.

    Returns:
        A tuple containing a list of Customer objects and the vehicle capacity.
    """
    customers = []
    capacity = 0  # Initialize vehicle capacity
    with open(file_path, 'r') as file:
        lines = file.readlines()
        vehicle_section = False
        customer_section = False
        for line in lines:
            # Check for section headers
            if line.startswith("VEHICLE"):
                vehicle_section = True
                customer_section = False
                continue
            elif line.startswith("CUSTOMER"):
                vehicle_section = False
                customer_section = True
                continue

            # Extract vehicle capacity
            if vehicle_section:
                if len(line.strip().split()) == 2 and line.strip().split()[0].isdigit():
                    capacity = int(line.strip().split()[1])

            # Extract customer data
            elif customer_section and len(line.strip().split()) == 7 and line.strip().split()[0].isdigit():
                cust_data = line.split()
                cust_no = int(cust_data[0])
                xcoord = float(cust_data[1])
                ycoord = float(cust_data[2])
                demand = int(cust_data[3])
                ready_time = int(cust_data[4])
                due_date = int(cust_data[5])
                service_time = int(cust_data[6])

                customer = Customer(cust_no, xcoord, ycoord, demand, ready_time, due_date, service_time)
                customers.append(customer)

    return customers, capacity

def calculate_distance(x1, y1, x2, y2):
    """
    Calculates the Euclidean distance between two points.

    Args:
        x1, y1: Coordinates of the first point.
        x2, y2: Coordinates of the second point.

    Returns:
        The calculated Euclidean distance.
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def create_distance_matrix(customers):
    """
    Creates a distance matrix for the given customers.

    Args:
        customers: A list of Customer objects.

    Returns:
        A matrix where element [i][j] represents the distance between customer i and j.
    """
    num_customers = len(customers)
    distance_matrix = [[0] * num_customers for _ in range(num_customers)]

    for i in range(num_customers):
        for j in range(num_customers):
            if i != j:
                distance_matrix[i][j] = calculate_distance(customers[i].xcoord, customers[i].ycoord,
                                                          customers[j].xcoord, customers[j].ycoord)
    return distance_matrix

def create_time_matrix(customers):
    """
    Creates a time matrix for the given customers.

    Args:
        customers: A list of Customer objects.

    Returns:
        A matrix where element [i][j] represents the travel time between customer i and j.
    """
    num_customers = len(customers)
    time_matrix = [[0] * num_customers for _ in range(num_customers)]

    for i in range(num_customers):
        for j in range(num_customers):
            if i != j:
                distance = calculate_distance(customers[i].xcoord, customers[i].ycoord,
                                              customers[j].xcoord, customers[j].ycoord)
                time_matrix[i][j] = distance / vehicle_speed

    return time_matrix

if __name__ == "__main__":
    csv_file_path = "./Benchmark/file_name.txt"
    customer_array = read_customers_from_txt(csv_file_path)   # Run the reading data process

