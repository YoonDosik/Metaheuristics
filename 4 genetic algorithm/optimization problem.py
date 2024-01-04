import math
import numpy as np
import random
import time


def f_x_y(x, y):  # 목적함수 정의

    '''
    x = x_1

    y = x_2

    value = 파이에 해당함

    '''

    value = math.pi  # 3.141592 ... --> 파이를 선언함

    return 21.5 + x * (math.sin(4 * value * x)) + y * (math.sin(20 * value * y))


# def Parameter_Constraints():

#     x = random.uniform(-3,12.1)           # X의 범위를 -3 ~ 12.1로 제한함
#     y = random.uniform(4.1,5.8)           # Y의 범위를 4.1 ~ 5.8로 제한함

#     return x,y


def Binary_Sampling(sample_num: float = None):
    '''

    sample_num = 몇개를 랜덤 추출 할지

    num_epoch = 만들어낼 데이터의 수

    '''

    a = np.array([int(0), int(1)])  # 0과 1로 구성된 배열을 생성함

    Binary = np.zeros([sample_num])  # 0과 1로 구성된 이진수가 생성됨

    for i in range(1):
        c = np.random.choice(a, sample_num, replace=True)  # sample_num = 33 --> 33자릿수를 생성하기 위해서
        Binary = c

    return Binary


def Demical_Number(Binary):
    x = Binary[:18]  # X는 앞에서 18번째 자릿수 까지만 사용함
    y = Binary[18:]  # Y는 18번째 자릿수 이후까지만 사용함

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

    return real_x, real_y


def Decoded_Solution(Binary):
    real_x, real_y = Demical_Number(Binary)  # 이진수를 십진수로 변경함

    x = -3.0 + (real_x * (12.1 + 3) / (2 ** 18 - 1))
    y = 4.1 + (real_y * (5.8 - 4.1) / (2 ** 15 - 1))

    return x, y


# =============================================================================
#            초기 랜덤 해를 생성하는 함수
# =============================================================================

def Initial_Population(sample_num: float = None):
    initial_encoding = Binary_Sampling(sample_num)

    x, y = Decoded_Solution(initial_encoding)

    initial_evaluation = np.zeros([1])

    initial_evaluation[0] = f_x_y(x, y)  # 초기 랜덤해의 목적함수 값

    return initial_evaluation, initial_encoding


# =============================================================================
#             pop_size만큼의 부모를 생성하는 함수
# =============================================================================


def Parents_make(pop_size: float = None, sample_num: float = None):
    Parents = []

    for i in range(pop_size):
        Binary = Binary_Sampling(sample_num)

        Parents.append(Binary)

    return Parents


# =============================================================================
#            Fitness를 계산하는 함수
# =============================================================================

def Parents_Evaluation(Parents, pop_size: float = None):
    Parents_Fitness = []

    for i in range(pop_size):
        x, y = Decoded_Solution(Parents[i])

        parents_evaluation = round(f_x_y(x, y), 5)  # 누적확률값을 사용하기 위해 소수점 5자리까지만 사용함

        Parents_Fitness.append(parents_evaluation)

    return Parents_Fitness


# =============================================================================
#                전역 최적해를 찾아주는 함수
# =============================================================================

def Global_Solution(initial_evaluation, Parents_Fitness, Parents, pop_size: float = None):
    global global_solution_value

    global_solution_value = np.zeros([1])

    global_solution_value[0] = initial_evaluation[0]

    for i in range(pop_size):

        if Parents_Fitness[i] >= global_solution_value[0]:
            global_solution_value[0] = Parents_Fitness[i]

            print("현재 최적해 값 -->", str(global_solution_value))

    return global_solution_value


# =============================================================================
#         Fitness에 대한 누적 확률 값을 계산하는 방법
# =============================================================================

def Parents_Prob(Parents_Fitness, pop_size: float = None):
    Parents_Probability = []

    for i in range(pop_size):
        Prob = Parents_Fitness[i] / sum(Parents_Fitness)
        Parents_Probability.append(Prob)

    Cumulative_Probability = (np.cumsum(Parents_Probability))

    return Parents_Probability, Cumulative_Probability


# =============================================================================
#            룰렛 방식에 의해 부모를 선택하는 함수
# =============================================================================

def Select_Function(Parents, Parents_Fitness, pop_size: float = None):
    Parents_Probability, Cumulative_Probability = Parents_Prob(Parents_Fitness, pop_size)

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


# =============================================================================
#                  Crossover를 진행하는 함수
# =============================================================================

def Crosssover_Operator(Select_Parents, Crossover_rate: float = None, pop_size: float = None):
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
            a = np.arange(0, 33, 1)

            Cut_point = np.random.choice(a, 1)  # 어디서 짜를지 랜덤으로 선택

            front_0 = Pair_Parents[i][0][:int(Cut_point)].astype(int)
            back_0 = Pair_Parents[i][0][int(Cut_point):].astype(int)

            front_1 = Pair_Parents[i][1][:int(Cut_point)].astype(int)
            back_1 = Pair_Parents[i][1][int(Cut_point):].astype(int)

            c = np.zeros([33]).astype(int)
            d = np.zeros([33]).astype(int)

            c[:int(Cut_point)] = front_0
            c[int(Cut_point):] = back_1

            d[int(Cut_point):] = back_0
            d[:int(Cut_point)] = front_1

            Cross_Parents.append([c, d])

        if cross_prob[i] >= Crossover_rate:
            e = np.zeros([1, 33]).astype(int)
            f = np.zeros([1, 33]).astype(int)

            e = Pair_Parents[i][0].astype(int)
            f = Pair_Parents[i][1].astype(int)

            Cross_Parents.append([e, f])

    return Cross_Parents


# =============================================================================
#               Mutation을 하는 함수
# =============================================================================

def Mutation_GA(Select_Parents, Crossover_rate: float = None, Mutation_rate: float = None, pop_size: float = None,
                sample_num: float = None):
    Cross_Parents = Crosssover_Operator(Select_Parents, Crossover_rate, pop_size)

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

                    if Cross_Parents[l][i][j] == int(0):

                        Cross_Parents[l][i][j] = int(1)

                    else:

                        Cross_Parents[l][i][j] = int(0)

    return Cross_Parents


# =============================================================================
#               기존의 부모 형태로 변형해주는 함수
# =============================================================================

def Return_Parents(Cross_Parents, pop_size: float = None):
    Update_Parents = []

    for i in range(int(pop_size / 2)):

        for j in range(2):
            Update_Parents.append(Cross_Parents[i][j])

    return Update_Parents


def Genetic_Algorithm(pop_size: float = None, sample_num: float = None, Crossover_rate: float = None,
                      Mutation_rate: float = None, epoch: float = None):
    initial_evaluation, initial_encoding = Initial_Population(sample_num)

    Parents = Parents_make(pop_size, sample_num)  # 초기 부모 생성

    Parents_Fitness = Parents_Evaluation(Parents, pop_size)

    global_solution_value = Global_Solution(initial_evaluation, Parents_Fitness, Parents, pop_size)

    Select_Parents = Select_Function(Parents, Parents_Fitness, pop_size)

    Cross_Parents = Mutation_GA(Select_Parents, Crossover_rate, Mutation_rate, pop_size, sample_num)

    Parents = Return_Parents(Cross_Parents, pop_size)

    for i in range(epoch):
        Parents_Fitness = Parents_Evaluation(Parents, pop_size)

        global_solution_value = Global_Solution(global_solution_value, Parents_Fitness, Parents, pop_size)

        Select_Parents = Select_Function(Parents, Parents_Fitness, pop_size)

        Cross_Parents = Mutation_GA(Select_Parents, Crossover_rate, Mutation_rate, pop_size, sample_num)

        Parents = Return_Parents(Cross_Parents, pop_size)

    print("최종 최적해는 : ", str(global_solution_value))

    return global_solution_value

