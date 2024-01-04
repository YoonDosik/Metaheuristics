import numpy as np
import random
import time
import math


def Word_Matching_Generation(sample_num: float = None):
    # Word_Matching_solution 생성
    '''
    sample_num = 몇개를 랜덤 추출 할지
    num_epoch = 만들어낼 데이터의 수
    '''

    a = np.arange(97, 123, 1)  # Word와 매칭되는 수의 배열을 생성함 EX) 97 --> a

    Word_Array = np.zeros([sample_num])  # 0과 1로 구성된 이진수가 생성됨

    for i in range(1):
        c = np.random.choice(a, sample_num, replace=True)  # sample_num = 33 --> 33자릿수를 생성하기 위해서
        Word_Array = c

    return Word_Array


def Decoded_Word(Word_Array):
    # 숫자를 해당하는 유니코드로 디코딩 하는 함수
    decoding_word = []

    for j in range(Word_Array.shape[0]):
        decoding_word.append(chr(Word_Array[j]))

    decoding_word = np.array(decoding_word)

    return decoding_word


def Parents_make(pop_size: float = None, sample_num: float = None):
    # 부모해를 생성하는 함수
    Parents = []

    for i in range(pop_size):
        Word_Array = Word_Matching_Generation(sample_num)

        Parents.append(Word_Array)

    return Parents


def Parents_Evaluation(Parents, pop_size: float = None):
    Parents_Decoding = []

    for i in range(pop_size):
        decoding_word = Decoded_Word(Parents[i])

        Parents_Decoding.append(decoding_word)

    return Parents_Decoding


def Word_Finess(Parents_Decoding, pop_size: float = None, sample_num: float = None):
    # Fitness를 계산하는 함수
    """
    정답인 a와 자리가 일치한다면 1, else --> 0으로 계산함
    """
    a = np.array(['t', 'o', 'b', 'e', 'o', 'r', 'n', 'o', 't', 't', 'o', 'b', 'e'])  # 정답 list
    Fitness = []
    for i in range(pop_size):
        for j in range(sample_num):
            c = Parents_Decoding[i][j] == a
            c = list(c)
            # 자릿수가 일치하는 것을 1로 카운트함
            fitness = c.count(True)
        Fitness.append(fitness)
    return Fitness


def Initial_Population(sample_num: float = None):
    # 초기해 평가 함수
    Word_Array = Word_Matching_Generation(sample_num)

    decoding_word = Decoded_Word(Word_Array)

    a = np.array(['t', 'o', 'b', 'e', 'o', 'r', 'n', 'o', 't', 't', 'o', 'b', 'e'])

    for i in range(sample_num):
        c = decoding_word[i] == a
        c = list(c)

    initial_evaluation = np.zeros([1])

    initial_evaluation[0] = c.count(True)

    return initial_evaluation


def Global_Solution(initial_evaluation, Parents_Fitness, pop_size: float = None):
    # 전역 최적해를 찾는 함수
    global global_solution_value

    global_solution_value = np.zeros([1])

    global_solution_value[0] = initial_evaluation[0]

    for i in range(pop_size):

        if Parents_Fitness[i] > global_solution_value[0]:
            global_solution_value[0] = Parents_Fitness[i]

            print("현재 최적해 값 -->", str(global_solution_value))

    return global_solution_value


def Parents_Prob(Fitness, pop_size: float = None):
    Parents_Probability = []

    for i in range(pop_size):
        Prob = Fitness[i] / sum(Fitness)
        Parents_Probability.append(Prob)

    Cumulative_Probability = (np.cumsum(Parents_Probability))

    return Parents_Probability, Cumulative_Probability


def Select_Function(Parents, Fitness, pop_size: float = None):
    Parents_Probability, Cumulative_Probability = Parents_Prob(Fitness, pop_size)

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


def Crosssover_Operator(Select_Parents, Crossover_rate: float = None, pop_size: float = None, sample_num: float = None):
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
            t = np.arange(0, sample_num, 1)

            Cut_point = np.random.choice(t, 1)  # 어디서 짜를지 랜덤으로 선택

            front_0 = Pair_Parents[i][0][:int(Cut_point)].astype(int)
            back_0 = Pair_Parents[i][0][int(Cut_point):].astype(int)

            front_1 = Pair_Parents[i][1][:int(Cut_point)].astype(int)
            back_1 = Pair_Parents[i][1][int(Cut_point):].astype(int)

            c = np.zeros([sample_num]).astype(int)
            d = np.zeros([sample_num]).astype(int)

            c[:int(Cut_point)] = front_0
            c[int(Cut_point):] = back_1

            d[:int(Cut_point)] = front_1
            d[int(Cut_point):] = back_0

            Cross_Parents.append([c, d])

        if cross_prob[i] >= Crossover_rate:
            e = np.zeros([1, sample_num]).astype(int)
            f = np.zeros([1, sample_num]).astype(int)

            e = Pair_Parents[i][0].astype(int)
            f = Pair_Parents[i][1].astype(int)

            Cross_Parents.append([e, f])

    return Cross_Parents


def Mutation_GA(Cross_Parents, Mutation_rate: float = None, pop_size: float = None, sample_num: float = None):
    Mutation_rate = 0.1

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
                    a = np.arange(97, 123, 1)

                    Cross_Parents[l][i][j] == np.random.choice(a, 1)  # 122는 z를 의미함 --> 1을 키우면 안되기 때문에 122일때만 1을 빼기로함

    return Cross_Parents


def Return_Parents(Cross_Parents, pop_size: float = None):
    Update_Parents = []

    for i in range(int(pop_size / 2)):

        for j in range(2):
            Update_Parents.append(Cross_Parents[i][j])

    return Update_Parents


def Word_Genetic_Algorithm(pop_size: float = None, sample_num: float = None, Crossover_rate: float = None,
                           Mutation_rate: float = None, epoch: float = None):
    initial_evaluation = Initial_Population(sample_num)

    Parents = Parents_make(pop_size, sample_num)

    Parents_Decoding = Parents_Evaluation(Parents, pop_size)

    Fitness = Word_Finess(Parents_Decoding, pop_size, sample_num)

    global_solution_value = Global_Solution(initial_evaluation, Fitness, pop_size)

    Select_Parents = Select_Function(Parents, Fitness, pop_size)

    Cross_Parents = Crosssover_Operator(Select_Parents, Crossover_rate, pop_size, sample_num)

    Cross_Parents = Mutation_GA(Cross_Parents, Mutation_rate, pop_size, sample_num)

    Parents = Return_Parents(Cross_Parents, pop_size)

    for i in range(epoch):
        Parents_Decoding = Parents_Evaluation(Parents, pop_size)

        Fitness = Word_Finess(Parents_Decoding, pop_size, sample_num)

        global_solution_value = Global_Solution(global_solution_value, Fitness, pop_size)

        Select_Parents = Select_Function(Parents, Fitness, pop_size)

        Cross_Parents = Crosssover_Operator(Select_Parents, Crossover_rate, pop_size, sample_num)

        Cross_Parents = Mutation_GA(Cross_Parents, Mutation_rate, pop_size, sample_num)

        Parents = Return_Parents(Cross_Parents, pop_size)

    print("최종 최적해는 : ", str(global_solution_value))

    return global_solution_value

