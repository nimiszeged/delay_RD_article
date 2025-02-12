import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.random import uniform, seed
from random import randint
import io


def main():
    delta = 0.01
    a = 0.1
    b = 0.9
    p = a + b
    r = (a + b) * (a + b)
    q = b / r
    tau = 0
    bound = 0.005
    d11 = 0.01
    d12 = 0.01
    d21 = 0.2
    d22 = 0.2
    Lx = 5
    Ly = 2


    gam_x = Lx * Lx
    gam_y = Ly * Ly
    du1 = d11 / gam_x
    du2 = d12 / gam_y
    dv1 = d21 / gam_x
    dv2 = d22 / gam_y
    n = 100
    m = 100
    dx = 1 / (n - 1)
    dy = 1 / (m - 1)
    dt = 0.00001
    de_steps = int(tau / dt)
    de_steps_2 = de_steps + 2

    u = np.random.uniform(low=p - bound, high=p + bound, size=(n, m, de_steps_2))
    v = np.random.uniform(low=q - bound, high=q + bound, size=(n, m, de_steps_2))

    time = int(500 / dt) + de_steps
    ik = 0

    print("print")
    ik = ik + 1
    file_name = "Rand" + str(ik) + ".txt"
    file = open(file_name, 'w')
    for i in range(1, n):
        for j in range(1, m):
            file.write("%i %i %f %f\n" % (i, j, u[i, j, 0], v[i, j, 0]))
        file.write("\n")
    file.close()

    ddux = du1 / (dx * dx)
    dduy = du2 / (dy * dy)
    ddvx = dv1 / (dx * dx)
    ddvy = dv2 / (dy * dy)

    for k in range(de_steps + 1, time):  # replace k-1 by k-tau where we introduce delay
        if (k - de_steps) % 1000 == 0:
            print("time: " + str((k - de_steps) * dt))
        kk = k % de_steps_2
        kkm1 = (k - 1) % de_steps_2
        kkdelay = (k - 1 - de_steps) % de_steps_2
        lst = -1
        uuv = u[1:n - 1, 1:m - 1, kkm1] * u[1:n - 1, 1:m - 1, kkm1] * v[1:n - 1, 1:m - 1, kkm1]
        u[1:n - 1, 1:m - 1, kk] = u[1:n - 1, 1:m - 1, kkm1] + \
                                  dt * (a - u[1:n - 1, 1:m - 1, kkm1] - 2 * uuv +
                                        3 * u[1:n - 1, 1:m - 1, kkdelay] * u[1:n - 1, 1:m - 1, kkdelay] *
                                        v[1:n - 1, 1:m - 1, kkdelay] +
                                        ddux * (u[2:n, 1:m - 1, kkm1] + u[0:n - 2, 1:m - 1, kkm1] - 2 *
                                                u[1:n - 1, 1:m - 1, kkm1]) +
                                        dduy * (u[1:n - 1, 2:m, kkm1] + u[1:n - 1, 0:m - 2, kkm1] - 2 *
                                                u[1:n - 1, 1:m - 1, kkm1]))

        v[1:n - 1, 1:m - 1, kk] = v[1:n - 1, 1:m - 1, kkm1] + \
                                  dt * (b - uuv +
                                        ddvx * (v[2:n, 1:m - 1, kkm1] + v[0:n - 2, 1:m - 1, kkm1] - 2 *
                                                v[1:n - 1, 1:m - 1, kkm1]) +
                                        ddvy * (v[1:n - 1, 2:m, kkm1] + v[1:n - 1, 0:m - 2, kkm1] - 2 *
                                                v[1:n - 1, 1:m - 1, kkm1]))

        u[0, 1:m - 1, kk] = u[1, 1:m - 1, kk]
        u[n - 1, 1:m - 1, kk] = u[n - 2, 1:m - 1, kk]
        v[0, 1:m - 1, kk] = v[1, 1:m - 1, kk]
        v[n - 1, 1:m - 1, kk] = v[n - 2, 1:m - 1, kk]

        u[:, 0, kk] = u[:, 1, kk]
        u[:, m - 1, kk] = u[:, m - 2, kk]
        v[:, 0, kk] = v[:, 1, kk]
        v[:, m - 1, kk] = v[:, m - 2, kk]


        if (k - de_steps) % 100000 == 0:
            print("print")
            ik = ik+1
            file_name = "Rand" + str(ik) + ".txt"
            file = open(file_name, 'w')
            for i in range(1, n):
                for j in range(1, m):
                    file.write("%i %i %f %f\n" % (i, j, u[i, j, kk], v[i, j, kk]))
                file.write("\n")
            file.close()

    print("the program has ended")


if __name__ == '__main__':
    main()

