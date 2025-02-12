import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.random import uniform, seed
from random import randint
import io

from statistics import mean
import ctypes
from ctypes import *
my_functions = CDLL("/home/nirmali/jobs/nirmali.so")

def compkernel(u, v, n, m, de_steps_2, dt, time, a, b, p, q, du1, du2, dv1, dv2, threshold, nc = 1):
    my_functions.compkernel.argtypes = [ np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"), np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"), \
      ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_double, ctypes.c_int64, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int64 ]
    my_functions.compkernel.restype = ctypes.POINTER(ctypes.c_double)
    ret = np.ctypeslib.as_array(my_functions.compkernel(np.reshape(u,nc*de_steps_2*n*m), np.reshape(v,nc*de_steps_2*n*m), n, m, de_steps_2, dt, time, a, b, p, q, du1, du2, dv1, dv2, threshold, nc), shape=(1,nc))
    if (nc == 1):
      return ret[0,0]
    else:
      return ret[0]

def getlastu(n, m, de_steps_2, nc = 1):
    my_functions.getlastu.argtypes = []
    my_functions.getlastu.restype = ctypes.POINTER(ctypes.c_double)
    return np.ctypeslib.as_array(my_functions.getlastu(), shape=(nc, de_steps_2,n,m))

def getlastv(n, m, de_steps_2, nc = 1):
    my_functions.getlastv.argtypes = []
    my_functions.getlastv.restype = ctypes.POINTER(ctypes.c_double)
    return np.ctypeslib.as_array(my_functions.getlastv(), shape=(nc, de_steps_2,n,m))

def main():
    nc = 10
    delta = 0.01
    a = 0.1
    b = 0.9
    # epsilon = np.sqrt(0.001)
    # eps1 = epsilon * epsilon
    # time = 10000000
    p = a + b
    r = (a + b) * (a + b)
    q = b / r
    tau = 0.2
    threshold = 0.01
    bound = 0.005
    d11 = 0.01
    d12 = 0.01
    d21 = 0.2
    d22 = 0.2
    Ly = 0.2

    lmin = 0.39
    lstep1 = 0.005
    lmax = 3
    length_lst = np.arange(lmin, lmax, lstep1)
    time_lst = []

    new_list = []
    for Lx in length_lst:
        print("Lx: " + str(Lx))
        gam_x = Lx * Lx
        gam_y = Ly * Ly
        du1 = d11 / gam_x
        du2 = d12 / gam_y
        dv1 = d21 / gam_x
        dv2 = d22 / gam_y
        n = max(int(Lx / delta), 50)
        m = 11
        dx = 1 / (n - 1)
        dy = 1 / (m - 1)
        dt = 1 / ((n - 1) ** 2 / Lx ** 2 + (m - 1) ** 2 / Ly ** 2)
        de_steps = int(tau / dt)
        de_steps_2 = de_steps + 2

        u = np.random.uniform(low=p - bound, high=p + bound, size=(nc, n, m, de_steps_2))
        v = np.random.uniform(low=q - bound, high=q + bound, size=(nc, n, m, de_steps_2))

        time = int(80 / dt) + de_steps

        min_time = mean(compkernel(u, v, n, m, de_steps_2, dt, time, a, b, p, q, du1/(dx*dx), du2/(dy*dy), dv1/(dx*dx), dv2/(dy*dy), threshold, nc))
        print("The min time is " + str(min_time))
        new_list.append(Lx)
        time_lst.append(min_time)

    file_name = "1_ttp_lx_3_tau_0.2_rand.txt"
    file = open(file_name, 'w')
    for i in range(np.size(new_list, 0)):
        file.write("%f %f\n" % (new_list[i], time_lst[i]))
    file.close()

    bif = 0.382088
    ylim = 12
    plt.figure()
    plt.plot(new_list, time_lst)
    plt.xlim(0, lmax)
    plt.ylim(0, 80)
    plt.xlabel('Lx')
    plt.ylabel('time')

    plt.title('Time to pattern!')
    # plt.plot([bif, bif], [0, ylim], '--')
    # plt.show()

    print("the program has ended")


if __name__ == '__main__':
    main()

    # parameter list 1
    # tn = 0.0
    # dt = 0.005
    # n = 101
    # m = 101 #6
    # Lx = np.sqrt(0.025)
    # Ly = np.sqrt(0.025)
    # dx = 1
    # dy = 1
    # a = 0.1
    # b = 0.9
    # epsilon = np.sqrt(0.025)
    # eps1 = epsilon * epsilon
    # gam_x = Lx * Lx
    # gam_y = Ly * Ly
    # du1 = eps1 / gam_x
    # du2 = eps1 / gam_y
    # dv1 = 1 / gam_x
    # dv2 = 1 / gam_y
