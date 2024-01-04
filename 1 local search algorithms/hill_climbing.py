from sympy import *
import numpy as np
import random
import time

def f_x_y(x, y):  # 원함수 선언

    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def Tweak(x, y):
    x_Step_wise = random.randint(-1, 1)
    y_Step_wise = random.randint(-1, 1)

    x_update = x + x_Step_wise
    y_update = y + y_Step_wise

    return x_update, y_update

def Hill_Climbing(Num_epoch: float = None):
    start = time.time()

    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)  # some initial candidate solution

    for i in range(Num_epoch):

        x_update, y_update = Tweak(x, y)

        if (f_x_y(x, y) > f_x_y(x_update, y_update)):
            x = x_update
            y = y_update

    print(f_x_y(x, y))
    print(str(x), str(y))
    print('time : ', time.time() - start)

    return x, y


x, y = Hill_Climbing(Num_epoch=100000)