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
    initial_p_, beta_u_, gamma_, mu_, inform_p_ = 0.2, 0.3, 0.5, 0.4, 0.4
    # joint parameters
    sigma_ = 0.5

    rho = 0.3
    alpha_ = 5

    m0_ = 0.6

    # global parameter
    max_step_, min_step_, min_err_ = 1000, 200, 1e-12
    # picture parameter
    point = 100
    repeat_time = 500
    correlation = ['0']
    # multiproess parameter
    pro_num_ = 5

    lamb_list = []
    for i in range(1, point):
        lamb_list += [1 * i / point]

    beta_u_list = []
    for i in range(point + 1):
        beta_u_list += [1 * i / point]
    print(beta_u_list)

    for corr in correlation:
        title = corr + '_new_musae_ENGB'
        graph_a, fname_a = us.read_graph(filename=title + '.txt', separator='\t')
        print(fname_a)

        title = corr + '_new_musae_ENGB_delete'
        graph_b, fname_b = us.read_graph(filename=title + '.txt', separator='\t')
        print(fname_b)

        mmca_mk_a_list1 = []
        mmca_mk_i_list1 = []

        node_list = list(graph_a.nodes())
        node_num = len(node_list)

        print(node_num)

        # informing a fraction of inform_p nodes according to the nodes degree rank
        mk_a_list, mk_i_list = us.mmca_of_uau_sis_mk_multi_process(pro_num=pro_num_,
                                                                       beta_list=beta_u_list,
                                                                       net_a=graph_a,
                                                                       net_b=graph_b,
                                                                       initial_p=initial_p_,
                                                                       inform_p=inform_p_,
                                                                       lamb=lamb_,
                                                                       gamma=gamma_,
                                                                       delta=delta_,
                                                                       sigma=sigma_, mu=mu_,
                                                                       rho_m=rho,
                                                                       m_0=m0_,
                                                                       alpha=alpha_,
                                                                       max_step=max_step_,
                                                                       min_step=min_step_,
                                                                       min_err=min_err_)
        print(str(beta_u_) + ':mmca multiprocess finished!')

        with open("0_check_theory_musaeENGB_a.txt", "w", encoding="utf-8") as f:
            # 保存为数组形式：[10, 20, 30, 40, 50]
            f.write(str(mk_a_list))

        with open("0_check_theory_musaeENGB_i.txt", "w", encoding="utf-8") as f:
            # 保存为数组形式：[10, 20, 30, 40, 50]
            f.write(str(mk_i_list))

        i_list_mc_mp = []
        a_list_mc_mp = []
        for beta_u_ in beta_u_list:
            ave_a_mp, ave_i_mp = us.mc_of_uau_sis_mk_multi_process(pro_num=pro_num_, repeat_time=repeat_time,
                                                                   net_a=graph_a, net_b=graph_b,
                                                                  initial_p=initial_p_, inform_p=inform_p_, lamb=lamb_,
                                                                       beta_u=beta_u_, beta_a=gamma_ * beta_u_,
                                                                       delta=delta_, sigma=sigma_,
                                                                       mu=mu_, rho_m=rho,alpha=alpha_,m_0=m0_,max_step=max_step_,
                                                                       min_step=min_step_, min_err=min_err_)
            print(str(beta_u_) + ':mc multiprocess finished!')
            i_list_mc_mp += [ave_i_mp]
            a_list_mc_mp += [ave_a_mp]


        with open("0_check_theory_musaeENGB_mca.txt", "w", encoding="utf-8") as f:
            # 保存为数组形式：[10, 20, 30, 40, 50]
            f.write(str(a_list_mc_mp))

        with open("0_check_theory_musaeENGB_mci.txt", "w", encoding="utf-8") as f:
            # 保存为数组形式：[10, 20, 30, 40, 50]
            f.write(str(i_list_mc_mp))


        # plt.plot(beta_u_list, mk_i_list, 'r-', label='mk_i')
        # plt.plot(beta_u_list, mk_a_list, 'y-', label='mk_a')
        #
        # # plt.plot(beta_u_list, o_i_list, 'b-', label='o_i')
        # # plt.plot(beta_u_list, o_a_list, 'r-', label='o_a')
        #
        # # plt.plot(beta_u_list, om_i_list, 'bo', label='om_i')
        # # plt.plot(beta_u_list, om_a_list, 'b*', label='om_a')
        #
        # # plt.plot(beta_u_list, mp_i_list_mmca, 'r-', label='ρI_mmca_mp')
        # # plt.plot(beta_u_list, mp_a_list_mmca, 'y-', label='ρA_mmca_mp')
        # #
        # plt.plot(beta_u_list, i_list_mc_mp, 'ro', label='ρI_mc_mp')
        # plt.plot(beta_u_list, a_list_mc_mp, 'y*', label='ρA_mc_mp')
        # plt.legend()
        # # plt.savefig(corr + 'checko_mu2_mp.png')
        # plt.savefig(corr + 'sigma'+str(sigma_)+'mi_m'+str(inform_p_)+'.png')
        # plt.close()
        # # plt.savefig('checko_mu2.png')
