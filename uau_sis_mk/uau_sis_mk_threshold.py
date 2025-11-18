import math
import time
import networkx as nx
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import uau_sis_mk_backup_functions as us
from numba import njit
from scipy import optimize
from collections import namedtuple

if __name__ == '__main__':

    start = time.time()

    # UAU paprameters
    lamb_, delta_ = 0.15, 0.6
    # SIOS parameters
    initial_p_, beta_u_, gamma_, mu_, inform_p_ = 0.2, 0.3, 0.6, 0.4, 0.4
    # joint parameters
    sigma_ = 0.5
    # 独有参数
    # rho_list = [0, 0.1, 0.3, 0.6, 1]
    # rho_list = [0.2]
    rho_ = 0.3
    # alpha_list = [1, 5, 10, 20, 30]
    alpha_ = 10
    m0_ = 0.6
    m0_list = [0, 0.6, 0.7, 0.8, 0.95]
    # global parameter
    max_step_, min_step_, min_err_ = 1000, 200, 1e-12
    # picture parameter
    point = 100
    repeat_time = 100
    correlation = ['0']

    inform_p_list = np.linspace(0,1,6)
    # correlation = ['0']
    # multiproess parameter
    pro_num_ = 5

    lamb_list = []
    for i in range(1, (point+1)):
        lamb_list += [1 * i / point]
    print(lamb_list)

    gamma_list = []
    for i in range(1 + point):
        gamma_list += [1 * i / point]
    print(gamma_list)

    beta_u_list = []
    for i in range(point + 1):
        beta_u_list += [1 * i / point]
    print(beta_u_list)

    for corr in correlation:
        thred_list = []
        title = corr + '_new_lastfm_asia'
        graph_a, fname_a = us.read_graph(filename=title + '.txt', separator='\t')
        print(fname_a)

        title = corr + '_new_lastfm_asia_delete'
        graph_b, fname_b = us.read_graph(filename=title + '.txt', separator='\t')
        print(fname_b)

        res_list = []

        for m0_ in m0_list:
            res_list1 = []
            for lamb_ in lamb_list:
                thred = us.threshold_uau_sis_mk(net_a=graph_a, net_b=graph_b,
                                                initial_p=initial_p_,
                                                lamb=lamb_,
                                                delta=delta_,
                                                gamma=gamma_,
                                                mu=mu_,
                                                rho_m=rho_,
                                                m_0=m0_,
                                                alpha=alpha_,
                                                max_step=max_step_,
                                                min_step=min_step_,
                                                min_err=min_err_)
                res_list1 += [thred]
                print(str(lamb_) + ':thread multiprocess finished!')
            print(res_list1)
            print(str(m0_) + ':thread multiprocess finished!')

            res_list += [res_list1]

        with open("lastfm_m0_thread.txt", "w", encoding="utf-8") as f:
            # 保存为数组形式：[10, 20, 30, 40, 50]
            f.write(str(res_list))

        # thred = us.threshold_uau_sis_mk(net_a=graph_a, net_b=graph_b,
        #                                 initial_p=initial_p_,
        #                                 lamb=lamb_,
        #                                 delta=delta_,
        #                                 gamma=gamma_,
        #                                 mu=mu_,
        #                                 rho_m=rho_,
        #                                 m_0=m0_,
        #                                 alpha=alpha_,
        #                                 max_step=max_step_,
        #                                 min_step=min_step_,
        #                                 min_err=min_err_)
        #
        # print(thred)