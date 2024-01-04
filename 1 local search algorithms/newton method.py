from sympy import *
import numpy as np
import random
import time

def Matrix_Derive_f_x_y(X):  # 각 편미분에 대한 함수 선언

    x, y = X

    return np.matrix([[-400 * x * (-x ** 2 + y) + 2 * (x - 1)], [200 * (y - x ** 2)]])


def Derive_2_f_x_y(X):  # 2차 편미분 함수 Matrix 선언

    x, y = X

    Derive_x_x = 1200 * x ** 2 - 400 * y + 2
    Derive_x_y = -400 * x
    Derive_y_x = -400 * x
    Derive_y_y = 200

    return np.matrix([[Derive_x_x, Derive_x_y], [Derive_y_x, Derive_y_y]])


def Newtons_Method(Num_epoch: float = None, Step_wise: float = None, epsilon: float = None):
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

        Descent = np.matmul(np.linalg.inv(Derive_2_f_x_y(X)), (Matrix_Derive_f_x_y(X)))

        Descent_arr = (np.asarray(Descent)).flatten()  # 행렬을 배열로 변환해주는 코드

        X = X - Step_wise * (Descent_arr)

        print(X)

        if np.linalg.norm(abs(Matrix_Derive_f_x_y(X))) < epsilon:
            break

    print('time : ', time.time() - start)
    print("초기점 :", str(x_init), "최적해 :", str(X), '반복횟수 :', str(i))

    return X, i


X, i = Newtons_Method(Num_epoch=10000, Step_wise=0.01, epsilon=0)