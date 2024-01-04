import math
import numpy as np
import random
import time
import pandas as pd


def Data_Matrix():
    # csv로 matrix를 생성한 후 불러옴
    Flow_data = np.array(pd.read_csv("C:/Users/com/Desktop/flow.csv", header=None))
    Distance_data = np.array(pd.read_csv("C:/Users/com/Desktop/distance.csv", header=None))

    return Flow_data, Distance_data


def Location_List():
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

    location_list = np.zeros([a.shape[0]])  # 0 ~ 14로 구성된 location 배열을 생성함

    for i in range(a.shape[0]):
        c = np.random.choice(a, a.shape[0], replace=False)

        location_list = c

    return location_list


def Calculation_Cost(Flow_data, Distance_data, location_list, sample_num: float = None):
    Cost = []

    for i in range(sample_num):

        a = []

        for j in range(sample_num):
            cost = (Flow_data[location_list[i] - 1][location_list[j] - 1]) * (Distance_data[i][j])
            a.append(cost)

        Cost.append(a)

    Cost = np.sum(Cost)

    return Cost


def Chromosome_make(pop_size: float = None):
    '''

    pop_size : 생성할 Chromosome의 갯수

    '''

    Parents = []

    for i in range(pop_size):
        location_list = Location_List()

        Parents.append(location_list)

    return Parents


def QAP_Fitness(Flow_data, Distance_data, Parents, pop_size: float = None, sample_num: float = None):
    Chromosome_Cost = []

    for i in range(pop_size):
        Cost = Calculation_Cost(Flow_data, Distance_data, Parents[i], sample_num)
        Chromosome_Cost.append(Cost)

    Fitness = []

    for i in range(pop_size):
        a = round((1 / Chromosome_Cost[i]) * 10000, 5)
        Fitness.append(a)

    return Chromosome_Cost, Fitness


def QAP_Initial_Population(Flow_data, Distance_data, pop_size: float = None, sample_num: float = None):
    Initial_Parents = Chromosome_make(pop_size)

    Chromosome_Cost = []

    for i in range(pop_size):
        Cost = Calculation_Cost(Flow_data, Distance_data, Initial_Parents[i], sample_num)

        Chromosome_Cost.append(Cost)

    initial_evaluation = np.min(Chromosome_Cost)

    index = np.argmin(initial_evaluation)

    initial_Elite = Initial_Parents[index]

    return initial_evaluation, initial_Elite


def QAP_Global_Solution(Parents, initial_Elite, initial_evaluation, Chromosome_Cost):
    if np.min(Chromosome_Cost) < initial_evaluation:

        global_solution = (np.min(Chromosome_Cost))

        index = np.argmin(Chromosome_Cost)

        Global_Chromosome = Parents[index]

        print("현 최적해 입니다... -->", str(global_solution))

    else:

        global_solution = initial_evaluation

        print("현 최적해 입니다... -->", str(global_solution))

        Global_Chromosome = initial_Elite

    return Global_Chromosome, global_solution


def QAP_Parents_Prob(Fitness, pop_size: float = None):
    Parents_Probability = []

    for i in range(pop_size):
        Prob = Fitness[i] / sum(Fitness)
        Parents_Probability.append(Prob)

    Cumulative_Probability = (np.cumsum(Parents_Probability))

    return Parents_Probability, Cumulative_Probability


def QAP_Select_Function(Parents, Cumulative_Probability, pop_size: float = None):
    Select_Prob = []  # pop_size 만큼의 랜덤확률을 생성함

    for i in range(pop_size):
        a = random.uniform(0, 1)  # 랜덤 선택 확률을 0~1의 범위로 생성함
        Select_Prob.append(a)

    Select_Parents = []  # 랜던 선택 확률에 의해 선택된 부모들

    for i in range(pop_size):

        for j in range(pop_size):

            if j == 0:

                if 0 <= Select_Prob[i] < Cumulative_Probability[0]:
                    Select_Parents.append(np.array(Parents[j]))

            else:

                if Cumulative_Probability[j - 1] <= Select_Prob[i] < Cumulative_Probability[j]:
                    Select_Parents.append(np.array(Parents[j]))

    return Select_Parents


def QAP_Crosssover_Operator(Select_Parents, Crossover_rate: float = None, pop_size: float = None,
                            sample_num: int = None):
    Pair_Parents = []  # 선택된 부모의 짝을 지어준 리스트

    for i in range(int(pop_size / 2)):
        j = i * 2  # slicing을 하기 위해 j 와 l 을 선언하여 슬라이싱을 진행함
        l = j + 2

        Pair_Parents.append(Select_Parents[j:l])

    cross_prob = []  # i번째 Pair Parents에 CrossOver가 일어날 확률

    for i in range(int(pop_size / 2)):
        a = random.uniform(0, 1)  # 랜덤 선택 확률을 0~1의 범위로 생성함
        cross_prob.append(a)

    Cross_Parents = []

    for i in range(int(pop_size / 2)):

        if cross_prob[i] < Crossover_rate:
            a = np.arange(0, sample_num, 1)

            Cut_point = np.random.choice(a, 1)  # 어디서 짜를지 랜덤으로 선택

            front_0 = Pair_Parents[i][0][:int(Cut_point)]
            back_0 = Pair_Parents[i][0][int(Cut_point):]

            front_1 = Pair_Parents[i][1][:int(Cut_point)]
            back_1 = Pair_Parents[i][1][int(Cut_point):]

            c = np.zeros([sample_num + 1])
            d = np.zeros([sample_num + 1])

            c[:int(Cut_point)] = front_0
            c[int(Cut_point):] = back_1

            d[:int(Cut_point)] = front_1
            d[int(Cut_point):] = back_0

            Cross_Parents.append([c, d])

        if cross_prob[i] >= Crossover_rate:
            e = np.zeros([1, sample_num])
            f = np.zeros([1, sample_num + 1])

            e = Pair_Parents[i][0]
            f = Pair_Parents[i][1]

            Cross_Parents.append([e, f])

    return Cross_Parents


def QAP_Mutation_GA(Select_Parents, Mutation_rate: float = None, pop_size: float = None, sample_num: float = None):
    Pair_Parents = []  # 선택된 부모의 짝을 지어준 리스트

    for i in range(int(pop_size / 2)):
        j = i * 2  # slicing을 하기 위해 j 와 l 을 선언하여 슬라이싱을 진행함
        l = j + 2

        Pair_Parents.append(Select_Parents[j:l])

    Mutation_Probability = []

    for l in range(int(pop_size / 2)):

        c = []

        for i in range(2):

            b = []

            for j in range(sample_num):
                a = round(random.uniform(0, 1), 5)
                b.append(a)

            c.append(b)

        Mutation_Probability.append(c)

    for l in range(int(pop_size / 2)):

        for i in range(2):

            for j in range(sample_num):

                if Mutation_Probability[l][i][j] < Mutation_rate:

                    if j != 0:

                        a = Pair_Parents[l][i][j]
                        b = Pair_Parents[l][i][j - 1]

                        Pair_Parents[l][i][j] = b
                        Pair_Parents[l][i][j - 1] = a

                    else:

                        c = Pair_Parents[l][i][j]
                        d = Pair_Parents[l][i][j + 1]

                        Pair_Parents[l][i][j] = d
                        Pair_Parents[l][i][j + 1] = c

    return Pair_Parents


def QAP_Return_Parents(Cross_Parents, pop_size: float = None):
    Update_Parents = []

    for i in range(int(pop_size / 2)):

        for j in range(2):
            Update_Parents.append(Cross_Parents[i][j])

    return Update_Parents


def QAP_Genetic_Algorithm(pop_size: float = None, sample_num: float = None, Crossover_rate: float = None,
                          Mutation_rate: float = None, epoch: float = None):
    Flow_data, Distance_data = Data_Matrix()

    initial_evaluation, initial_Elite = QAP_Initial_Population(Flow_data, Distance_data, pop_size, sample_num)

    global_solution = initial_evaluation  # 초기 전역 최적해 선언

    start = time.time()

    for i in range(epoch):
        Parents = Chromosome_make(pop_size)

        # Parents_Cost = QAP_Cost_Calculation(Flow_data, Distance_data, Parents, sample_num, pop_size)

        Chromosome_Cost, Fitness = QAP_Fitness(Flow_data, Distance_data, Parents, pop_size, sample_num)

        Global_Chromosome, global_solution = QAP_Global_Solution(Parents, initial_Elite, global_solution,
                                                                 Chromosome_Cost)

        Parents_Probability, Cumulative_Probability = QAP_Parents_Prob(Fitness, pop_size)

        Select_Parents = QAP_Select_Function(Parents, Cumulative_Probability, pop_size)

        # Cross_Parents = QAP_Crosssover_Operator(Select_Parents, Crossover_rate, pop_size, sample_num)

        Pair_Parents = QAP_Mutation_GA(Select_Parents, Mutation_rate, pop_size, sample_num)

        Parents = QAP_Return_Parents(Pair_Parents, pop_size)

    end = time.time()

    print("최종 최적해는 : ", str(global_solution))
    print("최적의 Chromosome : => " + str(Global_Chromosome))
    print("총 소요 시간 : ", f"{end - start:.5f} sec")

    return Global_Chromosome, global_solution
