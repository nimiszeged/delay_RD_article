import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.random import uniform, seed
from random import randint
import io


def ustar(a, b):
    return a + b


def vstar(a, b):
    return b / (a + b) ** 2


def jac_min_lam(a, b, tau, d11, d22, Lx, Ly, kx, ky, lam):
    us = ustar(a, b)
    vs = vstar(a, b)
    J = np.ndarray((2, 2))
    J[0, 0] = - d11 / Lx ** 2 * kx ** 2 * np.pi ** 2 - d11 / Ly ** 2 * ky ** 2 * np.pi ** 2 - 1 - 4 * us * vs + \
              6 * us * vs * np.exp(- tau * lam) - lam
    J[0, 1] = - 2 * us ** 2 + 3 * us ** 2 * np.exp(- tau * lam)
    J[1, 0] = - 2 * us * vs
    J[1, 1] = - d22 * kx ** 2 * np.pi ** 2 / Lx ** 2 - d22 * ky ** 2 * np.pi ** 2 / Ly ** 2 - us ** 2 - lam
    return J


def char(a, b, tau, d11, d22, Lx, Ly, kx, ky, lam):
    return np.linalg.det(jac_min_lam(a, b, tau, d11, d22, Lx, Ly, kx, ky, lam))


def d_char(a, b, tau, d11, d22, Lx, Ly, kx, ky, lam):
    elt = np.exp(- tau * lam)
    return 1 + a ** 2 + 2 * a * b + b ** 2 + 4 * b / (a + b) - 6 * b * elt / (a + b) +\
        d11 * kx ** 2 * np.pi ** 2 / Lx ** 2 + d22 * kx ** 2 * np.pi ** 2 / Lx ** 2 +\
        d11 * ky ** 2 * np.pi ** 2 / Ly ** 2 + d22 * ky ** 2 * np.pi ** 2 / Ly ** 2 +\
        2 * lam + 6 * b * d22 * elt * kx ** 2 * np.pi ** 2 * tau / (
            (a + b) * Lx ** 2) + 6 * b * d22 * elt * ky ** 2 * np.pi ** 2 * tau / (
            (a + b) * Ly ** 2) + 6 * b * elt * lam * tau / (a + b)


def newton(a, b, tau, d11, d22, Lx, Ly, kx, ky, lam, n_steps, tol):
    n = 0
    step_size = 2 * tol
    while (n < n_steps and step_size > tol):
        old_lam = lam
        lam = lam - char(a, b, tau, d11, d22, Lx, Ly, kx, ky, lam) / d_char(a, b, tau, d11, d22, Lx, Ly, kx, ky, lam)
        step_size = abs(lam - old_lam)
        n = n + 1
    return lam


def eigenvector(a, b, tau, d11, d22, Lx, Ly, kx, ky, lam):
    m = jac_min_lam(a, b, tau, d11, d22, Lx, Ly, kx, ky, lam)
    m[0, 0] = 0
    m[0, 1] = 1
    ev = np.linalg.solve(m, np.array([1, 0]))
    return 1 / np.max(np.abs(ev)) * ev


def main():
    delta = 0.01
    a = 0.1
    b = 0.9
    # epsilon = np.sqrt(0.001)
    # eps1 = epsilon * epsilon
    # time = 10000000
    p = a + b
    r = (a + b) * (a + b)
    q = b / r
    tau = 0
    threshold = 0.01
    bound = 0.005
    d11 = 0.01
    d12 = 0.01
    d21 = 0.2
    d22 = 0.2
    Ly = 0.2

    lmin = 0.39
    lstep1 = 0.005
    lmid = 0.4
    lstep2 = 0.005
    lmax = 3
    length_lst = np.append(np.arange(lmin, lmid, lstep1), np.arange(lmid, lmax + lstep2, lstep2))
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
        u_star_m = p * np.ones((n, m))
        v_star_m = q * np.ones((n, m))
        u = np.zeros((n, m, de_steps_2))
        v = np.zeros((n, m, de_steps_2))

        if Lx <= 0.929892:
            dom_mode_x = 1
        elif Lx <= 1.60033:
            dom_mode_x = 2
        elif Lx <= 2.25913:
            dom_mode_x = 3
        elif Lx <= 2.91436:
            dom_mode_x = 4
        else:
            dom_mode_x = 5
        dom_mode_y = 0

        lam = newton(a, b, tau, d11, d22, Lx, Ly, dom_mode_x, dom_mode_y, 0.5, 50, 0.00000000001)
        print("Dominant eigenvalue: " + str(lam))
        ev = eigenvector(a, b, tau, d11, d22, Lx, Ly, dom_mode_x, dom_mode_y, lam)
        print("Dominant eigenvector: " + str(ev))

        cx = np.zeros((n, 1))
        cx[:, 0] = np.cos(dom_mode_x * np.pi * np.linspace(0, 1, n))
        cy = np.zeros((m, 1))
        cy[:, 0] = np.cos(dom_mode_y * np.pi * np.linspace(0, 1, m))
        c = bound * cx * cy.transpose()

        for k in range(de_steps_2):
            t = - (de_steps_2 - 1 - k) * dt
            u[:, :, k] = u_star_m + np.exp(lam * t) * ev[0] * c
            v[:, :, k] = v_star_m + np.exp(lam * t) * ev[1] * c

        time = int(40 / dt) + de_steps

        for k in range(de_steps + 1, time):  # replace k-1 by k-tau where we introduce delay
            if (k - de_steps) % 1000 == 0:
                print("time: " + str((k - de_steps) * dt))
            kk = k % de_steps_2
            kkm1 = (k - 1) % de_steps_2
            kkdelay = (k - 1 - de_steps) % de_steps_2
            lst = -1
            u[1:n - 1, 1:m - 1, kk] = u[1:n - 1, 1:m - 1, kkm1] + \
                                      dt * (a - u[1:n - 1, 1:m - 1, kkm1] - 2 * u[1:n - 1, 1:m - 1, kkm1] *
                                            u[1:n - 1, 1:m - 1, kkm1] * v[1:n - 1, 1:m - 1, kkm1] +
                                            3 * u[1:n - 1, 1:m - 1, kkdelay] * u[1:n - 1, 1:m - 1, kkdelay] *
                                            v[1:n - 1, 1:m - 1, kkdelay]) + \
                                      du1 * dt * (u[2:n, 1:m - 1, kkm1] + u[0:n - 2, 1:m - 1, kkm1] - 2 *
                                                  u[1:n - 1, 1:m - 1, kkm1]) / (dx * dx) + \
                                      du2 * dt * (u[1:n - 1, 2:m, kkm1] + u[1:n - 1, 0:m - 2, kkm1] - 2 *
                                                  u[1:n - 1, 1:m - 1, kkm1]) / (dy * dy)

            v[1:n - 1, 1:m - 1, kk] = v[1:n - 1, 1:m - 1, kkm1] + \
                                      dt * (b - u[1:n - 1, 1:m - 1, kkm1] * u[1:n - 1, 1:m - 1, kkm1] *
                                            v[1:n - 1, 1:m - 1, kkm1]) + \
                                      dv1 * dt * (v[2:n, 1:m - 1, kkm1] + v[0:n - 2, 1:m - 1, kkm1] - 2 *
                                                  v[1:n - 1, 1:m - 1, kkm1]) / (dx * dx) + \
                                      dv2 * dt * (v[1:n - 1, 2:m, kkm1] + v[1:n - 1, 0:m - 2, kkm1] - 2 *
                                                  v[1:n - 1, 1:m - 1, kkm1]) / (dy * dy)

            u[0, 1:m - 1, kk] = u[1, 1:m - 1, kk]
            u[n - 1, 1:m - 1, kk] = u[n - 2, 1:m - 1, kk]
            v[0, 1:m - 1, kk] = v[1, 1:m - 1, kk]
            v[n - 1, 1:m - 1, kk] = v[n - 2, 1:m - 1, kk]

            u[:, 0, kk] = u[:, 1, kk]
            u[:, m - 1, kk] = u[:, m - 2, kk]
            v[:, 0, kk] = v[:, 1, kk]
            v[:, m - 1, kk] = v[:, m - 2, kk]

            l = max(np.max(np.absolute(u[:, :, kk] - u_star_m)), np.max(np.absolute(v[:, :, kk] - v_star_m)))
            if lst < l:
                lst = l

            if lst > threshold or k == time - 1:
                new_list.append(Lx)
                min_time = (k - de_steps) * dt
                time_lst.append(min_time)
                print("The min time is" + str(min_time))
                break

    file_name = "3_ttp_tau_1_lx_3.txt"
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
