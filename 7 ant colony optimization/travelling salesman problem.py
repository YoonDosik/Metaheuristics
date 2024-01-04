import numpy as np
import random

def Asymmetric_Matrix():
    # 거리 행렬을 생성함

    Distance_Matrix = np.array([[9999, 3, 5, 48, 48, 8, 8, 5, 5, 3, 3, 0, 3, 5, 8, 8, 5],
                                [3, 9999, 3, 48, 48, 8, 8, 5, 5, 0, 0, 3, 0, 3, 8, 8, 5],
                                [5, 3, 9999, 72, 72, 48, 48, 24, 24, 3, 3, 5, 3, 0, 48, 48, 24],
                                [48, 48, 74, 9999, 0, 6, 6, 12, 12, 48, 48, 48, 48, 74, 6, 6, 12],
                                [48, 48, 74, 0, 9999, 6, 6, 12, 12, 48, 48, 48, 48, 74, 6, 6, 12],
                                [8, 8, 50, 6, 6, 9999, 0, 8, 8, 8, 8, 8, 8, 50, 0, 0, 8],
                                [8, 8, 50, 6, 6, 0, 9999, 8, 8, 8, 8, 8, 8, 50, 0, 0, 8],
                                [5, 5, 26, 12, 12, 8, 8, 9999, 0, 5, 5, 5, 5, 26, 8, 8, 0],
                                [5, 5, 26, 12, 12, 8, 8, 0, 9999, 5, 5, 5, 5, 26, 8, 8, 0],
                                [3, 0, 3, 48, 48, 8, 8, 5, 5, 9999, 0, 3, 0, 3, 8, 8, 5],
                                [3, 0, 3, 48, 48, 8, 8, 5, 5, 0, 9999, 3, 0, 3, 8, 8, 5],
                                [0, 3, 5, 48, 48, 8, 8, 5, 5, 3, 3, 9999, 3, 5, 8, 8, 5],
                                [3, 0, 3, 48, 48, 8, 8, 5, 5, 0, 0, 3, 9999, 3, 8, 8, 5],
                                [5, 3, 0, 72, 72, 48, 48, 24, 24, 3, 3, 5, 3, 9999, 48, 48, 24],
                                [8, 8, 50, 6, 6, 0, 0, 8, 8, 8, 8, 8, 8, 50, 9999, 0, 8],
                                [8, 8, 50, 6, 6, 0, 0, 8, 8, 8, 8, 8, 8, 50, 0, 9999, 8],
                                [5, 5, 26, 12, 12, 8, 8, 0, 0, 5, 5, 5, 5, 26, 8, 8, 9999]])

    return Distance_Matrix


def Tau_Matrix(initial_tau):
    global tau_matrix

    tau_matrix = np.zeros(shape=(17, 17))

    for i in range(17):

        for j in range(17):
            tau_matrix[i, j] = initial_tau

    return tau_matrix


def Probabilty_Matrix(Distance_Matrix, alpha, beta):
    global Probability_matrix, tau_matrix

    a = np.zeros(shape=(17, 17))

    for i in range(17):

        for j in range(17):
            c = np.power(tau_matrix[i, j], alpha)

            d = np.power((1 / Distance_Matrix[i, j]), beta)

            a[i, j] = c * d

            # inf라면 0이 아닌 1의 값으로 처리함

    for i in range(len(a)):
        a[i][np.isinf(a[i])] = 1

    Probability_matrix = a

    return Probability_matrix


def Creation_Start(ant_count):
    # 총 목적지의 index

    a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    # ant_count만큼 선택을 진행함

    initial_index = np.random.choice(a, ant_count)

    Ant = []

    for i in range(ant_count):
        Ant.append([])

    for i in range(ant_count):
        Ant[i].append(initial_index[i])

    return Ant


def Creation_Solution(Distance_Matrix, ant_count):
    global Probability_matrix, tau_matrix

    Ant = Creation_Start(ant_count)

    for l in range(len(Ant)):

        b = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

        for i in range(17):

            b.remove(Ant[l][i])

            Sum = []

            for q in b:
                c = Probability_matrix[Ant[l][i], q]

                Sum.append(c)

            Probabliity = []

            for j in range(len(Sum)):
                Probabliity.append(Sum[j] / np.sum(Sum))

            Cumulative_Probability = (np.cumsum(Probabliity))

            Select_Prob = random.uniform(0, 1)  # 랜덤확률을 생성함

            for j in range(len(b)):

                if j == 0:

                    if 0 <= Select_Prob < Cumulative_Probability[0]:
                        Ant[l].append(b[0])

                else:

                    if Cumulative_Probability[j - 1] <= Select_Prob < Cumulative_Probability[j]:
                        Ant[l].append(b[j])

    return Ant


def Create_solution_q(Distance_Matrix, ant_count, q_0):
    global Probability_matrix, tau_matrix

    Ant = Creation_Start(ant_count)

    for l in range(len(Ant)):

        b = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

        for i in range(17):

            b.remove(Ant[l][i])

            Sum = []

            for q in b:
                c = Probability_matrix[Ant[l][i], q]

                Sum.append(c)

            Probablilty = []

            for j in range(len(Sum)):
                Probablilty.append(Sum[j] / np.sum(Sum))

            Cumulative_Probability = (np.cumsum(Probablilty))

            Random_Prob = random.uniform(0, 1)

            for i in Probablilty:

                if len(Probablilty) > 0:
                    index = np.argmax(i)
                    Max_element = b[index]

            if Random_Prob < q_0:

                Ant[l].append(Max_element)

            else:

                Select_Prob = random.uniform(0, 1)  # 랜덤확률을 생성함

                for j in range(len(b)):

                    if j == 0:

                        if 0 <= Select_Prob < Cumulative_Probability[0]:
                            Ant[l].append(b[0])

                    else:

                        if Cumulative_Probability[j - 1] <= Select_Prob < Cumulative_Probability[j]:
                            Ant[l].append(b[j])

    return Ant


def Initial_Evaluate_Ant(Distance_Matrix, Ant):
    Cost_Ant = []

    for i in range(len(Ant)):

        a = []

        for j in range(len(Ant[i])):

            if j != 16:

                a.append(Distance_Matrix[(Ant[i][j]), (Ant[i][j + 1])])

            else:

                a.append(Distance_Matrix[(Ant[i][j]), (Ant[i][0])])

        Cost_Ant.append(np.sum(a))

    Best_Ant_Cost = min(Cost_Ant)

    Best_Ant_Index = np.argmin(Cost_Ant)

    Best_Ant = Ant[Best_Ant_Index]

    return Best_Ant_Cost, Best_Ant, Cost_Ant


def Delta_Tau(Cost_Ant, Ant):
    Inverse_Cost_Ant = []

    for i in range(len(Cost_Ant)):
        Inverse_Cost_Ant.append(1 / Cost_Ant[i])

    Del_tau = []

    for i in range(len(Ant)):

        a = np.zeros(shape=(17, 17))

        for j in range(len(Ant[i])):

            if j == 16:

                a[Ant[i][16], Ant[i][0]] = Inverse_Cost_Ant[i]

            else:

                a[Ant[i][j], Ant[i][j + 1]] = Inverse_Cost_Ant[i]

        Del_tau.append(a)

    Delta_tau = sum(Del_tau)

    return Delta_tau


def Update_Tau(rho, Delta_tau):
    global tau_matrix

    New = (rho) * tau_matrix + (1 - rho) * Delta_tau

    tau_matrix = New

    return tau_matrix


def Best_ANT(Distance_Matrix, Best_Ant_Cost, Best_Ant, Ant):
    Cost_Ant = []

    for i in range(len(Ant)):

        a = []

        for j in range(len(Ant[i])):

            if j == 16:

                a.append(Distance_Matrix[Ant[i][16], Ant[i][0]])

            else:

                a.append(Distance_Matrix[Ant[i][j], Ant[i][j + 1]])

        Cost_Ant.append(np.sum(a))

    for i in range(len(Cost_Ant)):

        if Best_Ant_Cost <= Cost_Ant[i]:

            Best_Ant_Cost = Best_Ant_Cost

        else:

            Best_Ant_Cost = Cost_Ant[i]
            Best_Ant_Index = np.argmin(Cost_Ant[i])
            Best_Ant = Ant[Best_Ant_Index]

    return Best_Ant, Best_Ant_Cost, Cost_Ant


def ACO(ant_count, alpha, beta, rho, iterations):
    global tau_matrix, Probability_matrix

    # 초기 타우값은 0.008

    Distance_Matrix = Asymmetric_Matrix()

    # alpha = 1, beta = 2

    tau_matrix = Tau_Matrix(0.008)

    Probability_matrix = Probabilty_Matrix(Distance_Matrix, alpha, beta)

    Ant = Creation_Solution(Distance_Matrix, ant_count)

    Best_Ant_Cost, Best_Ant, Cost_Ant = Initial_Evaluate_Ant(Distance_Matrix, Ant)

    for i in range(iterations):
        print(str(i), '--> 현재', str(i), '번째 Iteration 진행중 입니다.')

        Ant = Creation_Solution(Distance_Matrix, ant_count)

        Best_Ant, Best_Ant_Cost, Cost_Ant = Best_ANT(Distance_Matrix, Best_Ant_Cost, Best_Ant, Ant)

        Delta_tau = Delta_Tau(Cost_Ant, Ant)

        tau_matrix = Update_Tau(rho, Delta_tau)

        Probability_matrix = Probabilty_Matrix(Distance_Matrix, alpha, beta)

        print("Best_Ant_Cost는 : ", str(Best_Ant_Cost))

        print("Best Ant :", str(Best_Ant))

    return Best_Ant_Cost, Best_Ant, Cost_Ant, tau_matrix, Probability_matrix
