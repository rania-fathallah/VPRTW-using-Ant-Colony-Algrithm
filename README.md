# VPRTW using Ant Colony Algrithm


This project focused on addressing the Vehicle Routing Problem with Time Windows (VRPTW) by implementing an Ant Colony Optimization (ACO) algorithm. The ACO was customized to effectively handle the specific constraints of the VRPTW and aimed to find satisfactory solutions within reduced computational time.


The Vehicle Routing Problem with Time Windows (VRPTW) is a complex optimization
    problem that involves planning the routes of a fleet of vehicles to serve a set
    of customers. Each customer has a specific demand and a time window during which
    they must be served. The objective is to minimize the total distance traveled by
    the vehicles while satisfying all customer demands and time window constraints.

Key characteristics of VRPTW:

    Multiple vehicles: A fleet of vehicles is available to serve customers.
    Customers with demands: Each customer requires a specific quantity of goods.
    Time windows: Each customer specifies a time interval during which they can be served.
    Vehicle capacity: Each vehicle has a limited capacity to carry goods.
    Depot: All vehicles start and end their routes at a central depot.

Challenges of VRPTW:

    Combinatorial explosion: The number of possible routes increases exponentially with the number of customers.
    Time window constraints: Meeting customer time windows while optimizing routes is challenging.
    Vehicle capacity constraints: Ensuring that vehicles do not exceed their capacity.


Ant Colony Optimization (ACO) is a metaheuristic algorithm often used to solve VRPTW
due to its ability to handle complex constraints and find high-quality solutions. By
simulating the behavior of ants searching for food, ACO can effectively explore the
solution space and improve over time.


The results i got from this project are shown on the presentation 'VPRTW+ACS.pdf' at the secodn to last slide. And the code is detailed more there too.




