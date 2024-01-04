import numpy as np
import random
import math
import time
import pandas as pd


def F_x_y(x, y):  # 목적함수 정의

    return ((4 - (2.1 * x ** 2) + (x ** 4 / 3)) * x ** 2 + (x * y) + (-4 + (4 * y ** 2)) * y ** 2)


def Parameter_Constraints(Population_size: float = None):
    Population = []

    for i in range(Population_size):
        x = random.uniform(-3, 3)  # X의 제약조건
        y = random.uniform(-2, 2)  # Y의 제약조건

        a = [x, y]

        Population.append(a)

    return Population


def Random_Vector():
    r1 = random.uniform(0, 1)
    r2 = random.uniform(0, 1)

    return r1, r2


def Initiallize_Personal_Best(Population, Population_size: float = None):
    P_Best = []

    for i in range(Population_size):
        a = F_x_y(Population[i][0], Population[i][1])

        P_Best.append(a)

    return P_Best


def Initiallize_Global_Best(Population, P_Best, Population_size: float = None):
    a = np.arange(3, Population_size + 3, 3)

    Pop = []
    b = []

    for i in a:
        Pop.append(Population[i - 3: i])
        b.append(P_Best[i - 3: i])

    Global_Best = []
    Global_Solution = []

    for i in range(len(Pop)):
        Global_Best.append(b[i][np.argmin(b[i])])

        Global_Solution.append(Pop[i][np.argmin(b[i])])

    return Global_Solution, Global_Best


def Initial_Velocity(Population_size: float = None):
    velo = []

    for i in range(Population_size):

        a = []

        for j in range(2):
            a.append(random.uniform(-1, 1))

        velo.append(a)

    return velo


# =============================================================================
#
# =============================================================================

def Velocity(velo, Population, New_Population, Global_Best, Global_Solution, Population_size: float = None,
             C_1: float = None, C_2: float = None):
    a = np.arange(3, Population_size + 3, 3)

    r_1, r_2 = Random_Vector()

    New_Pop = []
    velocity = []
    population = []
    new_population = []

    for i in a:
        New_Pop.append(New_Population[i - 3: i])
        velocity.append(velo[i - 3: i])
        population.append(Population[i - 3: i])
        new_population.append(New_Population[i - 3: i])

    for i in range(len(Global_Solution)):

        for l in range(3):

            for j in range(2):
                velocity[i][l][j] = velocity[i][l][j] + (
                            C_1 * r_1 * (population[i][l][j] - new_population[i][l][j])) + (
                                                C_2 * r_2 * (Global_Solution[i][j] - new_population[i][l][j]))

    velocity = sum(velocity, [])

    return velocity


def Position_Update(velocity, New_Population, Population_size: float = None):
    for i in range(Population_size):

        for j in range(2):
            New_Population[i][j] = New_Population[i][j] + velocity[i][j]

    return New_Population


def Personal_Best(New_P_Best, P_Best, Population, New_Population, Population_size: float = None):
    for i in range(Population_size):

        if P_Best[i] < New_P_Best[i]:

            P_Best[i] = P_Best[i]

            Population[i] = Population[i]

        else:

            P_Best[i] = New_P_Best[i]

            Population[i] = New_Population[i]

    return P_Best, Population


def Function_Global_Best(Global_Solution, Global_Best, P_Best, New_Population, Population_size: float = None):
    a = np.arange(3, Population_size + 3, 3)

    Pop = []
    b = []

    for i in a:
        Pop.append(New_Population[i - 3: i])
        b.append(P_Best[i - 3: i])

    New_G_Best = []
    New_G_Solution = []

    for i in range(len(Pop)):
        New_G_Best.append(np.min(b[i]))

        New_G_Solution.append(Pop[i][np.argmin(b[i])])

    for i in range(len(Global_Best)):

        if Global_Best[i] < New_G_Best[i]:

            Global_Best[i] = Global_Best[i]

            Global_Solution[i] = Global_Solution[i]

        else:

            Global_Best[i] = New_G_Best[i]

            Global_Solution[i] = New_G_Solution[i]

    return Global_Best, Global_Solution


def Ring_Particle_Swarm_Optimization(epoch: float = None, Population_size: float = None, C_1: float = None,
                                     C_2: float = None):
    # 초기값들을 설정함

    population = Parameter_Constraints(Population_size)

    P_Best = Initiallize_Personal_Best(population, Population_size)

    Global_Solution, Global_Best = Initiallize_Global_Best(population, P_Best, Population_size)

    velocity = Initial_Velocity(Population_size)

    for i in range(epoch):
        New_Population = Parameter_Constraints(Population_size)

        velocity = Velocity(velocity, population, New_Population, Global_Best, Global_Solution, Population_size, C_1,
                            C_2)

        New_Population = Position_Update(velocity, New_Population, Population_size)

        New_P_Best = Initiallize_Personal_Best(New_Population, Population_size)

        P_Best, population = Personal_Best(New_P_Best, P_Best, population, New_Population, Population_size)

        Global_Best, Global_Solution = Function_Global_Best(Global_Solution, Global_Best, P_Best, population,
                                                            Population_size)

    Global_Best = round(np.min(Global_Best), 5)

    Global_Solution = Global_Solution[np.argmin(Global_Best)]

    print("최적해는 :", str(Global_Best), "Solution은", str(Global_Solution))

    return Global_Best, Global_Solution

