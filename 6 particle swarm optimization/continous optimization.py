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


def Initiallize_Global_Best(Population, Initial_P_Best, Population_size: float = None):
    Global_Solution = Population[np.argmin(Initial_P_Best)]
    Global_Best = np.min(Initial_P_Best)

    return Global_Solution, Global_Best


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
    New_G_Best = np.min(P_Best)

    New_Population[np.argmin(P_Best)]

    if Global_Best < New_G_Best:

        Global_Best = Global_Best

        Global_Solution = Global_Solution

    else:

        Global_Best = New_G_Best

        Global_Solution = New_Population[np.argmin(P_Best)]

    return Global_Best, Global_Solution


def Initial_Velocity(Population_size: float = None):
    velocity = []

    for i in range(Population_size):

        a = []

        for j in range(2):
            a.append(random.uniform(-1, 1))

        velocity.append(a)

    return velocity


def Velocity(velocity, Population, New_Population, Global_Best, Global_Solution, Population_size: float = None,
             C_1: float = None, C_2: float = None, Inertia_Weight: float = None):
    r_1, r_2 = Random_Vector()

    k = random.uniform(0, 1)

    pi = C_1 * r_1 + C_2 * r_2

    Constriction_Coefficient = 2 * k / abs(2 - pi - (pi * (pi - 4)) ** (1 / 2))

    for i in range(Population_size):

        for j in range(2):
            velocity[i][j] = Constriction_Coefficient * (
                        Inertia_Weight * velocity[i][j] + (C_1 * r_1 * (Population[i][j] - New_Population[i][j])) + (
                            C_2 * r_2 * (Global_Solution[j] - New_Population[i][j])))

    return velocity


def Position_Update(velocity, New_Population, Population_size: float = None):
    for i in range(Population_size):

        for j in range(2):
            New_Population[i][j] = New_Population[i][j] + velocity[i][j]

    return New_Population


def Particle_Swarm_Optimization(epoch: float = None, Population_size: float = None, C_1: float = None,
                                C_2: float = None, Inertia_Weight: float = None):
    # 초기값들을 설정함

    Population = Parameter_Constraints(Population_size)

    P_Best = Initiallize_Personal_Best(Population, Population_size)

    Global_Solution, Global_Best = Initiallize_Global_Best(Population, P_Best, Population_size)

    velocity = Initial_Velocity(Population_size)

    for i in range(epoch):
        New_Population = Parameter_Constraints(Population_size)

        velocity = Velocity(velocity, Population, New_Population, Global_Best, Global_Solution, Population_size, C_1,
                            C_2, Inertia_Weight)

        New_Population = Position_Update(velocity, New_Population, Population_size)

        New_P_Best = Initiallize_Personal_Best(New_Population, Population_size)

        P_Best, Population = Personal_Best(New_P_Best, P_Best, Population, New_Population, Population_size)

        Global_Best, Global_Solution = Function_Global_Best(Global_Solution, Global_Best, P_Best, Population,
                                                            Population_size)

    print("최적해는 :", str(Global_Best), "Solution은", str(Global_Solution))

    return Global_Best, Global_Solution

