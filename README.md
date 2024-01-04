# **Adaptive Optimization, Fall 2021**

### **1. Local Search Algorithms**

Based on the below well-known Rosenbrck function, try to apply several search algorithms and find the optimal solution using three local search algorithms. 

min f(x,y)=(1-x)^2+〖100(y-x^2 )〗^2
 
![image](https://github.com/YoonDosik/Metaheuristics/assets/144199897/cc4038d3-ca97-4e30-a138-d7552a607439)

	1. Gradient Ascent (Decent) Search
	2. Newton method
	3. Hill-Climbing

After apply three algorithms using Python, compare the performance among the algorithms.

-------------

### **2. Local Search Algorithms**

1) natural real solution representation and 2) binary encoded solution representation 

The numerical example of unconstrained optimization problem in Chap 1.2 from Gen & Cheng book is given as follows:

![image](https://github.com/YoonDosik/Metaheuristics/assets/144199897/ee107995-e356-4aaa-a206-c35940a86bbb)

	1. Define two kinds of “neighbor-hood” using in your algorithm. Then, apply those to 2).
	2. Develop a neighbor-hood local search (NH-LS) algorithm using 1) natural real solution representation and 2) binary encoded solution representation.

-------------

### **4. Genetic Algorithm**

1) Optimization problem

![image](https://github.com/YoonDosik/Metaheuristics/assets/144199897/e7d2e1aa-dcbb-4807-a2a9-6720f9aae216)

![image](https://github.com/YoonDosik/Metaheuristics/assets/144199897/630efb35-a84b-4c9c-a83b-27c94b3a259f)

2) Word matching problem

![image](https://github.com/YoonDosik/Metaheuristics/assets/144199897/6bf77ba1-0e5e-44e5-bdce-8c6f65eb95bb)

* Genetic Algorithms and Engineering Design. Mitsuo Gen and Runwei ChengCopyright © 1997 John Wiley & Sons, Inc2

-------------

### **5. Genetic Algorithm**

1) Quadratic Assignment Problem

This is the sixth of the QAP (Quadratic Assignment Problem) test problems of Negent et al.*. Fifteen departments are to be placed in 15 locations with five columns in three rows. The objective is to minimize flow costs between the placed departments. The flow cost is (flow × distance), where the both flow and distance are symmetric between any given pair of departments. Below is the flow (lower half) between departments and rectilinear distance (upper half) between location matrix. The optimal solution is 575 (or 1150 if you double the flows).     
* C. E. Nugent, T. E. Vollmann, and J. Ruml, An experimental comparison of techniques for the assignment of facilities to locations. Operations Research 16, 150-173 (1968)

![image](https://github.com/YoonDosik/Metaheuristics/assets/144199897/d58171e5-28cf-4a04-874f-0dce4d59af6b)

	To solve this problem, i had made the matrix related to distance and location

2) Continous Optimization

This is the six hump camelback function with two decision variables where x lies between ±3 and y lies between ±2. The objective is to minimize z. The global minimum lies at (-0.0898, 0.7126) where z = -1.0316.

![image](https://github.com/YoonDosik/Metaheuristics/assets/144199897/3c656be2-6aca-420b-9aa7-d1ad8c2ed774)


