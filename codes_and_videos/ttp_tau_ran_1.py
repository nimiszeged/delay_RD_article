import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.random import uniform, seed
from random import randint
import io
from statistics import mean
import time as mytime
import ctypes
from multiprocessing import Process
from ctypes import *
my_functions = CDLL("./nirmali.so")

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
#    rng = np.random.default_rng(seed=12280752884038410067234213672332424406)
    rng = np.random.default_rng()
    nc = 15 # Number of cases
#    u = rng.uniform(-1,1,size=(2,5,5))
#    print(u)
#    uu = printret(u,5,5,2)
#    print(uu)
#    print(u-uu)
#    return
    delta = 0.01
    a = 0.1
    b = 0.9
    # epsilon = np.sqrt(0.001)
    # eps1 = epsilon * epsilon
    # time = 10000000
    p = a + b
    r = (a + b) * (a + b)
    q = b / r
    threshold = 0.01
    bound = 0.005
    d11 = 0.01
    d12 = 0.01
    d21 = 0.2
    d22 = 0.2
    Ly = 0.1
    Lx = 1.2

    lmin = 0.0
    lstep1 = 0.005
    lmid = 0.4
    lstep2 = 0.005
    lmax = 1

    n = max(int(Lx / delta), 50)
    m = 11
    dx = 1 / (n - 1)
    dy = 1 / (m - 1)
    dt = 1 / ((n - 1) ** 2 / Lx ** 2 + (m - 1) ** 2 / Ly ** 2)
    gam_x = Lx * Lx
    gam_y = Ly * Ly
    du1 = d11 / gam_x / (dx * dx)
    du2 = d12 / gam_y / (dy * dy)
    dv1 = d21 / gam_x / (dx * dx)
    dv2 = d22 / gam_y / (dy * dy)

    tau_lst = np.append(np.arange(lmin, lmid, lstep1), np.arange(lmid, lmax + lstep2, lstep2))
    time_lst = []
    new_list = []
    for tau in tau_lst:
        print("tau: " + str(tau))
        de_steps = int(tau / dt)
        de_steps_2 = de_steps + 2
        u = rng.uniform(low=p - bound, high=p + bound, size=(nc, de_steps_2, n, m))
        v = rng.uniform(low=q - bound, high=q + bound, size=(nc, de_steps_2, n, m))
        time = int(200 / dt) + de_steps
        start_time = mytime.time()
        min_time = mean(compkernel(u, v, n, m, de_steps_2, dt, time, a, b, p, q, du1, du2, dv1, dv2, threshold, nc))
        print("The min time is " + str(min_time))
        print("The elapsed time is " + str(mytime.time()-start_time))
        new_list.append(tau)
        time_lst.append(min_time)
    file_name = "ttp_tau_ran_1.txt"
    file = open(file_name, 'w')
    for i in range(np.size(new_list, 0)):
        file.write("%f %f\n" % (new_list[i], time_lst[i]))
    file.close()

    bif = 0.382088
    ylim = 12
    plt.figure()
    plt.plot(new_list, time_lst)
    plt.xlim(0, lmax)
    plt.ylim(0, 12)
    plt.xlabel('Lx')
    plt.ylabel('time')

    plt.title('Time to pattern!')
    plt.plot([bif, bif], [0, ylim], '--')
    plt.show()

    print("the program has ended")

if __name__ == '__main__':
    p = Process(main())
    p.start()
    p.join()

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
