import math
import numpy as np
import random
import time


# 목적함수 정의
def f_x_y(x, y):
    '''
    x = x_1
    y = x_2
    value = 파이에 해당함
    '''
    value = math.pi

    return 21.5 + x * (math.sin(4 * value * x)) + y * (math.sin(20 * value * y))

# 목적 함수의 제약 조건
def Parameter_Constraints():
    x = random.uniform(-3, 12.1)
    y = random.uniform(4.1, 5.8)

    return x, y

def Tweak(x, y):
    while True:

        x_Step_wise = random.randint(-1, 1)
        y_Step_wise = random.randint(-1, 1)

        x_update = round(x + x_Step_wise, 5)  # 소수점 자릿수 5번째 자리로 설정함
        y_update = round(y + y_Step_wise, 5)  # 소수점 자릿수 5번째 자리로 설정함

        if ((x_update >= -3 and x_update <= 12.1) and (y_update >= 4.1 and y_update <= 5.8)):

            break
        else:

            print("다시 계산중... ")

            continue

    return x_update, y_update

def Binary_Sampling(sample_num: float = None, num: float = None):
    '''

    sample_num = 몇개를 랜덤 추출 할지

    num_epoch = 만들어낼 데이터의 수

    '''

    a = np.array([0, 1])

    Binary = []

    for i in range(num):
        c = np.random.choice(a, sample_num, replace=True)
        Binary.append(c)

    return Binary


def Demical_Number(Binary):
    x = np.array(Binary)[:, :18]
    y = np.array(Binary)[:, 18:]

    two_x = []

    for i in range(x.shape[1]):
        two_x.append(2 ** i)

    two_y = []

    for i in range(y.shape[1]):
        two_y.append(2 ** i)

    two_x = np.sort(np.array(two_x))[::-1]  # 내림차순으로 정렬
    two_y = np.sort(np.array(two_y))[::-1]

    real_x = np.sum(two_x * x)  # 2의 배수 배열과 x의 곱을 한후 합을 계산함
    real_y = np.sum(two_y * y)  # 2의 배수 배열과 y의 곱을 한후 합을 계산함

    return real_x, real_y


def Decoded_Solution(Binary):
    real_x, real_y = Demical_Number(Binary)

    x = -3.0 + (real_x * (12.1 + 3) / (2 ** 18 - 1))
    y = 4.1 + (real_y * (5.8 - 4.1) / (2 ** 15 - 1))

    return x, y


def Binary_Tweak_1(Binary):
    a = np.arange(0, 33, 1)  # 0 ~ 33의 정수를 생성함
    b = np.arange(0, 33, 1)

    random_num = np.random.choice(b, 1)  # 총 몇개를 바꿀 것인지를 랜덤으로 추출
    sample_index_num = np.random.choice(a, random_num)  # 몇 번째 index를 바꿀 것인지

    for j in range(len(Binary)):
        for i in range(len(sample_index_num)):
            if Binary[j][sample_index_num[i]] == 0:
                Binary[j][sample_index_num[i]] = 1
            else:
                Binary[j][sample_index_num[i]] = 0

    return Binary


def Binary_Tweak_2(Binary):

    Binary_update = []
    for i in range(len(Binary)):
        a = np.random.permutation(Binary[i])
        Binary_update.append(a)

    return Binary_update


def Encoding_Tweak1_NN_LC(sample_num: float = None, num: float = None, epoch: float = None):
    start = time.time()

    Binary = Binary_Sampling(sample_num, num)

    x, y = Decoded_Solution(Binary)

    for i in range(epoch):

        Binary = Binary_Tweak_1(Binary)
        x_update, y_update = Decoded_Solution(Binary)

        print(str(i), "번째 진행중입니다 ...")

        if (f_x_y(x, y) < f_x_y(x_update, y_update)):  # Maximize 문제이므로 목적함수에 대한 결과값이 더 커야함

            x = x_update
            y = y_update

    print("최적해의 결과 => " + str(f_x_y(x, y)))
    print("최적해 X1 :", str(x), "최적해 X2 :", str(y))
    print('총 소요 시간 time : ', time.time() - start)

    return x, y


def Encoding_Tweak2_NN_LC(sample_num: float = None, num: float = None, epoch: float = None):

    start = time.time()
    Binary = Binary_Sampling(sample_num, num)

    x, y = Decoded_Solution(Binary)

    for i in range(epoch):

        Binary_update = Binary_Tweak_2(Binary)
        x_update, y_update = Decoded_Solution(Binary_update)
        print(str(i), "번째 진행중입니다 ...")
        if (f_x_y(x, y) < f_x_y(x_update, y_update)):  # Maximize 문제이므로 목적함수에 대한 결과값이 더 커야함

            x = x_update
            y = y_update

    print("최적해의 결과 => " + str(f_x_y(x, y)))
    print("최적해 X1 :", str(x), "최적해 X2 :", str(y))
    print('총 소요 시간 time : ', time.time() - start)

    return x, y

Binary_Tweak_1_x, Binary_Tweak_1_y = Encoding_Tweak1_NN_LC(sample_num=33, num=1, epoch=1500)



