from sympy import *
import numpy as np
import random
import time

def Derive_f_x_y(X):  # 각 편미분에 대한 함수 선언

    x, y = X

    return np.array([-400 * x * (-x ** 2 + y) + 2 * (x - 1), 200 * (y - x ** 2)])


def Gradient_Descent(Num_epoch: float = None, Step_wise: float = None, epsilon: float = None):

    '''
    초기점 선언 하는 함수
    범위를 너무 확장 시키면 Nan이 나옴  --> 발산 하는 문제가 발생함
    Therfore 범위를 제한 시켜줌
    '''

    start = time.time()

    x = random.uniform(0, 1)
    y = random.uniform(0, 1)

    x_init = ([x, y])

    X = x_init

    '''
    Num_epoch : 얼만큼 반복 할 것인지에 대한 변수

    Step_wise : alpha에 해당함 --> 얼만큼 이동 시킬 건지에 대한 변수

    epsilon : 0에 근접한 한계선을 선언함
    '''

    for i in range(Num_epoch):

        X = X - Step_wise * (Derive_f_x_y(X))

        print(X)

        if np.linalg.norm(abs(Derive_f_x_y(X))) <= epsilon:
            break

    print('time : ', time.time() - start)
    print("초기점 :", str(x_init), "최적해 :", str(X), '반복횟수 :', str(i))

    return X, i


X, i = Gradient_Descent(Num_epoch=100000, Step_wise=0.001, epsilon=0.0001)
