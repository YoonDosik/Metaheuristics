
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


def Initial_NN_LC(epoch: float = None):

    start = time.time()
    x, y = Parameter_Constraints()

    for i in range(epoch):

        x_update, y_update = Tweak(x, y)

        print(str(i), "번째 진행중입니다 ...")

        if (f_x_y(x, y) < f_x_y(x_update, y_update)):  # Maximize 문제이므로 목적함수에 대한 결과값이 더 커야함

            x = x_update
            y = y_update

    print("최적해의 결과 => " + str(f_x_y(x, y)))
    print("최적해 X1 :", str(x), "최적해 X2 :", str(y))
    print('총 소요 시간 time : ', time.time() - start)

    return x, y

x,y = Initial_NN_LC(epoch = 1500)
