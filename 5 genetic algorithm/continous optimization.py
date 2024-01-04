import math
import numpy as np
import random
import time


def f_x_y(x, y):  # 목적함수 정의

    return ((4 - (2.1 * x ** 2) + (x ** 4 / 3)) * x ** 2 + (x * y) + (-4 + (4 * y ** 2)) * y ** 2)


def Binary_Sampling(sample_num: float = None):
    '''

    sample_num = Chromosome의 사이즈를 결정함 --> 32

    num_epoch = 만들어낼 데이터의 수 --> 1로 설정함

    '''

    a = np.array([0, 1])  # 0과 1로 구성된 배열을 생성함

    Binary = np.zeros([sample_num])  # 0과 1로 구성된 이진수가 생성됨

    for i in range(1):
        c = np.random.choice(a, sample_num, replace=True)  # sample_num = 33 --> 33자릿수를 생성하기 위해서
        Binary = c

    return Binary


def Demical_Number(Binary, slicing_num: int = None):
    '''

    slicing_num : 어디에서 자를지를 나타내는 변수

    '''

    x = Binary[:slicing_num]  # X는 앞에서 18번째 자릿수 까지만 사용함
    y = Binary[slicing_num:]  # Y는 18번째 자릿수 이후까지만 사용함

    two_x = []  # X에 대한 2의 배수 배열을 생성함

    for i in range(x.shape[0]):
        two_x.append(2 ** i)

    two_y = []

    for i in range(y.shape[0]):  # Y에 대한 2의 배수 배열을 생성함

        two_y.append(2 ** i)

    two_x = np.sort(np.array(two_x))[::-1]  # 내림차순으로 정렬
    two_y = np.sort(np.array(two_y))[::-1]  # 내림차순으로 정렬

    real_x = np.sum(two_x * x)  # 2의 배수 배열과 x의 곱을 한후 합을 계산함
    real_y = np.sum(two_y * y)  # 2의 배수 배열과 y의 곱을 한후 합을 계산함

    x = -3 + (real_x * (3 + 3) / (2 ** 16 - 1))
    y = -2 + (real_y * (2 + 2) / (2 ** 16 - 1))

    return x, y


def Parents_make(pop_size: float = None, sample_num: float = None):
    '''

    pop_size : 생성할 Chromosome의 갯수

    '''

    Parents = []

    for i in range(pop_size):
        Binary = Binary_Sampling(sample_num)

        Parents.append(Binary)

    return Parents


def Parents_Evaluation(Parents, pop_size: float = None, slicing_num: int = None):
    Parents_evaluation = []

    for i in range(pop_size):
        x, y = Demical_Number(Parents[i], slicing_num)

        parents_evaluation = round(f_x_y(x, y), 5)  # 소수점 5자리까지만 사용함

        Parents_evaluation.append(parents_evaluation)

    return Parents_evaluation


def Initial_Population(pop_size: float = None, sample_num: float = None, slicing_num: int = None):
    Initial_Parents = []

    for i in range(pop_size):
        Binary = Binary_Sampling(sample_num)

        Initial_Parents.append(Binary)

    initial_parents_evaluation = []

    for i in range(pop_size):
        x, y = Demical_Number(Initial_Parents[i], slicing_num)

        evaluation = round(f_x_y(x, y), 5)

        initial_parents_evaluation.append(evaluation)

    initial_evaluation = np.min(initial_parents_evaluation)

    index = np.argmin(initial_parents_evaluation)

    initial_Elite = Initial_Parents[index]

    return initial_evaluation, initial_Elite


def Global_Solution(Parents, initial_Elite, initial_evaluation, Parents_evaluation):
    if np.min(Parents_evaluation) < initial_evaluation:

        global_solution = (np.min(Parents_evaluation))

        index = np.argmin(Parents_evaluation)

        Global_Parents = Parents[index]

        print("현 최적해 입니다... -->", str(global_solution))

    else:

        global_solution = initial_evaluation

        print("현 최적해 입니다... -->", str(global_solution))

        Global_Parents = initial_Elite

    return Global_Parents, global_solution


def Parents_Prob(Parents_Evaluation, pop_size: float = None):
    Parents_Probability = []

    for i in range(pop_size):
        Prob = Parents_Evaluation[i] / sum(Parents_Evaluation)
        Parents_Probability.append(Prob)

    Cumulative_Probability = (np.cumsum(Parents_Probability))

    return Parents_Probability, Cumulative_Probability


def Select_Function(Parents, Cumulative_Probability, pop_size: float = None):
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


def Crosssover_Operator(Select_Parents, Crossover_rate: float = None, pop_size: float = None, sample_num: int = None):
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

            c = np.zeros([sample_num])
            d = np.zeros([sample_num])

            c[:int(Cut_point)] = front_0
            d[int(Cut_point):] = back_1

            c[:int(Cut_point)] = front_1
            d[int(Cut_point):] = back_0

            Cross_Parents.append([c, d])

        if cross_prob[i] >= Crossover_rate:
            e = np.zeros([1, sample_num])
            f = np.zeros([1, sample_num])

            e = Pair_Parents[i][0]
            f = Pair_Parents[i][1]

            Cross_Parents.append([e, f])

    return Cross_Parents


def Mutation_GA(Cross_Parents, Mutation_rate: float = None, pop_size: float = None, sample_num: float = None):
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

                    if Cross_Parents[l][i][j] == 0:

                        Cross_Parents[l][i][j] = 1

                    else:

                        Cross_Parents[l][i][j] = 0

    return Cross_Parents


def Return_Parents(Cross_Parents, pop_size: float = None):
    Update_Parents = []

    for i in range(int(pop_size / 2)):

        for j in range(2):
            Update_Parents.append(Cross_Parents[i][j])

    return Update_Parents


def Genetic_Algorithm(pop_size: float = None, sample_num: float = None, Crossover_rate: float = None,
                      Mutation_rate: float = None, slicing_num: int = None, epoch: float = None):
    initial_evaluation, initial_Elite = Initial_Population(pop_size, sample_num, slicing_num)

    global_solution = initial_evaluation  # 초기 전역 최적해 선언

    start = time.time()

    for i in range(epoch):
        Parents = Parents_make(pop_size, sample_num)

        Parents_evaluation = Parents_Evaluation(Parents, pop_size, slicing_num)

        Global_Parents, global_solution = Global_Solution(Parents, initial_Elite, global_solution, Parents_evaluation)

        Parents_Probability, Cumulative_Probability = Parents_Prob(Parents_evaluation, pop_size)

        Select_Parents = Select_Function(Parents, Cumulative_Probability, pop_size)

        Cross_Parents = Crosssover_Operator(Select_Parents, Crossover_rate, pop_size, sample_num)

        Cross_Parents = Mutation_GA(Cross_Parents, Mutation_rate, pop_size, sample_num)

        Parents = Return_Parents(Cross_Parents, pop_size)

    end = time.time()

    x, y = Demical_Number(Global_Parents, slicing_num)

    print("최적의 X : ", str(x), "최적의 Y :", str(y))
    print("최종 최적해는 : ", str(global_solution))
    print("최적의 Chromosome : => " + str(Global_Parents))
    print("총 소요 시간 : ", f"{end - start:.5f} sec")

    return Global_Parents, global_solution

