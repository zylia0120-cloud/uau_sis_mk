import math
import time
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from mpmath.rational import mpq_0
from numba import njit
from scipy import optimize
from collections import namedtuple
import operator
import scipy.stats as stats
from multiprocessing.pool import Pool



# def read_graph(filename=None, separator=None):
#     fname = filename
#     graph = nx.read_edgelist(fname, delimiter=separator, nodetype=int)
#     graph = graph.to_undirected()
#     print('is_connected:', nx.is_connected(graph))
#     return graph, fname.split('/')[-1]

def read_graph(filename=None, separator=None):
    fname = filename
    graph = nx.read_edgelist(fname, delimiter=' ', nodetype=float)
    # print('预备删边网络的大小',graph)
    graph = graph.to_undirected()
    return graph, fname.split('/')[-1]

def get_nei_dic(net=None):
    node_list = net.nodes()
    adjlist = {}
    for temp in node_list:
        adjlist[temp] = [n for n in net.neighbors(temp)]
    return adjlist


def del_dic_b_from_dic_a(dic_a=None, dic_b=None):
    for key in dic_b:
        if key in dic_a:
            dic_a.pop(key)
    return dic_a


def check_nodes_equality(net_a=None, net_b=None):
    # 检查网络中的节点是否一一对应
    nodes_a = net_a.nodes()
    nodes_b = net_b.nodes()
    print('the two networks have the same node list?', operator.eq(nodes_a, nodes_b))


def check_edges_equality(net_a=None, net_b=None):
    # 检查网络中的节点是否一一对应
    edges_a = net_a.edges()
    edges_b = net_b.edges()
    print('the two networks have the same edge list?', operator.eq(edges_a, edges_b))


def calculate_interlayer_degree_correlation(net_a=None, net_b=None):
    # calculate the correlation,due to the existence of nodes with the same degree,the corr should not be exact 1 or -1
    node_list = net_a.nodes()
    x = []
    y = []
    for i in node_list:
        x = x + [net_a.degree(i)]
        y = y + [net_b.degree(i)]
        # print(i,ER_a.degree(i))
    corr = stats.spearmanr(x, y)
    return corr


def calculate_average_degree(net=None):
    # calculate the average degree
    isolated_node_number = nx.degree_histogram(net)[0]
    node_num = len(net.nodes())
    print('the isolated node number is :', isolated_node_number)
    degree_list = nx.degree_histogram(net)[1:]  # remove the nodes with degree 0
    max_degree = len(degree_list)
    avg_k = 0
    Pk_dic = {}
    for k in range(1, max_degree + 1):
        if degree_list[k - 1] != 0:
            Pk_dic[k] = degree_list[k - 1] / node_num
            avg_k = avg_k + k * Pk_dic[k]
    return avg_k


def check_multinet_validity(net_a=None, net_b=None):
    check_nodes_equality(net_a=net_a, net_b=net_b)
    check_edges_equality(net_a=net_a, net_b=net_b)
    ave_a = calculate_average_degree(net=net_a)
    ave_b = calculate_average_degree(net=net_b)
    correlation = calculate_interlayer_degree_correlation(net_a=net_a, net_b=net_b)
    print('the average degree of net_a is', ave_a)
    print('the average degree of net_b is', ave_b)
    print('the interlayer degree correlation is', correlation)


# the following functions are used for UAU-mk-SIS model, which contains the disease by informing the hub nodes
def degree_rank_dic(net=None):
    node_list = net.nodes()
    degree_dic = {}
    for node in node_list:
        degree_dic[node] = net.degree(node)
    degree_list = sorted(degree_dic.items(), key=lambda degree_dic: degree_dic[1])
    for i in range(len(node_list)):
        degree_dic[degree_list[i][0]] = i + 1
    return degree_dic


# theory
def mmca_of_uau_sis_mk_for_multi_process(net_a, net_b, initial_p, inform_p, lamb, beta_u, beta_a, delta,
                                         sigma, mu, rho_m, alpha, m_0, max_step, min_step, min_err):
    """
        修改后的MMCA函数，在UAU层中引入自媒体个体

        新增参数:
        - media_nodes: 自媒体节点列表，如果为None则随机选择部分节点作为自媒体
        - me_base: 基础媒体环境感知概率
        - me_factor: 媒体环境感知因子，用于计算 me = me_base + me_factor * ρA
        """

    # initial states
    joint_dic = {}  # store the states probability of nodes
    node_list = list(net_a.nodes())
    node_num = len(node_list)

    degree_dic = degree_rank_dic(net_a)
    rank_c = int(node_num * inform_p)

    lamb_1 = 1 - (1 - lamb) ** alpha

    # 初始化自媒体节点,随机选择20%的节点作为自媒体
    media_num = int(node_num * rho_m)
    media_nodes = random.sample(node_list, media_num)

    # 创建自媒体节点标识字典
    is_media_node = {node: (node in media_nodes) for node in node_list}

    for node in node_list:
        joint_dic[node] = {'AI': 0, 'UI': 0, 'AS': 0, 'US': 1}

    # initial seeds
    seed_list = []
    seed_num = int(node_num * initial_p)

    while len(seed_list) <= seed_num:
        seed_index = random.randint(0, node_num - 1)  # the range is from 0 to node_num-1 included
        seed = node_list[seed_index]
        # print('node list', node_list)
        # print('seed index', seed_index)
        # print('seed', seed)
        if seed not in seed_list:
            seed_list = seed_list + [seed]
            joint_dic[seed] = {'AI': 0, 'UI': 1, 'AS': 0, 'US': 0}
    # print('seed list:', seed_list)

    # states evolution starts
    neigbor_dic_a = get_nei_dic(net_a)
    neigbor_dic_b = get_nei_dic(net_b)  # for calculating the complex transmission probablity
    temp_i = 0
    temp_a = 0
    fac_list_s, fac_list_i, fac_list_u, fac_list_a, time_list = [], [], [], [], []

    for t in range(max_step):
        # calculate the fraction of nodes in each state
        fac_s, fac_i, fac_u, fac_a = 0, 0, 0, 0

        fac_a_media = 0  # A态自媒体节点比例 ρA
        media_node_count = len(media_nodes)

        state_dic = {}
        for node in node_list:
            state_s = joint_dic[node]['AS'] + joint_dic[node]['US']
            fac_s = fac_s + state_s
            state_i = joint_dic[node]['AI'] + joint_dic[node]['UI']
            fac_i = fac_i + state_i
            state_u = joint_dic[node]['UI'] + joint_dic[node]['US']
            fac_u = fac_u + state_u
            state_a = joint_dic[node]['AI'] + joint_dic[node]['AS']
            fac_a = fac_a + state_a

            # 计算A态自媒体节点比例
            if is_media_node[node]:
                fac_a_media += state_a

            state_dic[node] = {'S': state_s, 'I': state_i, 'U': state_u, 'A': state_a}

        fac_s = fac_s / node_num
        fac_i = fac_i / node_num
        fac_u = fac_u / node_num
        fac_a = fac_a / node_num

        # 计算A态自媒体节点比例 ρA
        rho_a = fac_a_media / media_node_count if media_node_count > 0 else 0
        me = rho_a * m_0

        # print(me)

        fac_list_s.append(fac_s)
        fac_list_i.append(fac_i)
        fac_list_u.append(fac_u)
        fac_list_a.append(fac_a)
        time_list.append(t)

        # check whether the model is nomalized
        if (fac_a + fac_u - 1.0) > min_err or (fac_s + fac_i - 1.0) > min_err:
            print('The model is not normalized')
            print('the value of fac_a+fac_u is:', fac_a + fac_u)
            print('the value of fac_s+fac_i+fac_o is:', fac_a + fac_u)
            break
        # save time
        if (fac_i - temp_i) < min_err and (fac_a - temp_a) < min_err and (t > min_step):
            # print('complete ahead of schedule at step:', t)
            # print(fac_a, fac_list_a[-1], fac_i, fac_list_i[-1], fac_o, fac_list_o[-1])
            return (fac_a, fac_i)
        temp_i = fac_i
        temp_a = fac_a

        # calculate the complex transmission probablity of nodes
        complex_state_dic = {}  # store the complex transmission probablity of nodes
        for node in node_list:
            theta_i, qu_i, qa_i = 1, 1, 1
            for j in neigbor_dic_a[node]:
                theta_i = theta_i * (1 - (1 - rho_m) * state_dic[j]['A'] * lamb - rho_m * state_dic[j]['A'] * lamb_1)  # the node in U state did not receive message from all the neighbors
            for j in neigbor_dic_b[node]:
                qu_i = qu_i * (1 - state_dic[j]['I'] * beta_u)
                qa_i = qa_i * (1 - state_dic[j]['I'] * beta_a)
            complex_state_dic[node] = {'theta': theta_i, 'qa': qa_i, 'qu': qu_i}

            # update the states in t+1 according to the states in t
        for node in node_list:
            p_ai = joint_dic[node]['AI']
            p_ui = joint_dic[node]['UI']
            p_as = joint_dic[node]['AS']
            p_us = joint_dic[node]['US']
            theta = complex_state_dic[node]['theta']
            qa = complex_state_dic[node]['qa']
            qu = complex_state_dic[node]['qu']

            new_ui = 1.0 * (p_ui * theta * (1 - me) * (1 - mu) * (1 - sigma) +
                            p_ai * delta * (1 - me) * (1 - mu) * (1 - sigma) +
                            p_us * theta * (1 - me) * (1 - qu) * (1 - sigma) +
                            p_as * delta * (1 - me) * (1 - qu) * (1 - sigma)
                            )

            new_ai = 1.0 * (
                        p_ui * ((1 - theta) * (1 - mu) + theta * (1 - me) * (1 - mu) * sigma + theta * me * (1 - mu)) +
                        p_ai * ((1 - delta) * (1 - mu) + delta * (1 - me) * (1 - mu) * sigma + delta * me * (1 - mu)) +
                        p_us * (theta * me* (1 - qa) + theta * (1 - me) * (1 - qu) * sigma + (1 - theta) * (1 - qa)) +
                        p_as * (delta * me * (1 - qa) + delta * (1 - me) * (1 - qu) * sigma + (1 - delta) * (1 - qa))
                        )

            new_us = p_ui * theta * (1 - me) * mu + p_ai * delta * (1 - me) * mu + p_us * theta * (
                        1 - me) * qu + p_as * delta * (1 - me) * qu

            new_as = 1.0 * (p_ui * ((1 - theta) * mu + theta * me * mu) +
                            p_ai * ((1 - delta) * mu + delta * me * mu) +
                            p_us * ((1 - theta) * qa + theta * me * qa) +
                            p_as * ((1 - delta) * qa + delta * me * qa)
                            )

            joint_dic[node]['AI'] = new_ai
            joint_dic[node]['UI'] = new_ui
            joint_dic[node]['AS'] = new_as
            joint_dic[node]['US'] = new_us

            # # 计算A态自媒体节点比例
            # if is_media_node[node]:
            #     fac_a_media = fac_a_media + 1
            #
            # # 计算A态自媒体节点比例 ρA
            # rho_a = fac_a_media / media_node_count if media_node_count > 0 else 0
            # me = rho_a * m_0

        # fac_i = fac_list_i[-1]
        # fac_a = fac_list_a[-1]
        # fac_o = fac_list_o[-1]
        # print(fac_a, fac_list_i[-1], fac_i, fac_list_a[-1], fac_o,fac_list_o[-1])
    return (fac_a, fac_i)


def threshold_uau_sis_mk(net_a=None, net_b=None, initial_p=None, lamb=None, delta=None, gamma=None,
                         mu=None, rho_m=None, m_0=None, alpha=None, max_step=None, min_step=None, min_err=None):
    # 自媒体相关参数（与MMCA和MC函数保持一致）
    lamb_1 = 1 - (1 - lamb) ** alpha  # 自媒体传播概率

    # initial pa_dic
    pa_dic = {}  # store the probability of nodes in a-state
    node_list = list(net_a.nodes())
    node_num = len(node_list)

    # 初始化自媒体节点,随机选择20%的节点作为自媒体
    media_num = int(node_num * rho_m)
    media_nodes = random.sample(node_list, media_num)

    # 创建自媒体节点标识字典
    is_media_node = {node: (node in media_nodes) for node in node_list}

    for node in node_list:
        pa_dic[node] = 0

    # initial seeds
    seed_list = []
    seed_num = int(node_num * initial_p)
    while len(seed_list) <= seed_num:
        seed_index = random.randint(0, node_num - 1)  # the range is from 0 to node_num-1 included
        seed = node_list[seed_index]  # be careful about the index and the node-label
        # print('node list', node_list)
        # print('seed index', seed_index)
        # print('seed', seed)
        if seed not in seed_list:
            seed_list = seed_list + [seed]
            pa_dic[seed] = 1
    # print('seed list:', seed_list)

    rho_a = 0
    for i in node_list:
        rho_a += pa_dic[i]
    rho_a = rho_a / node_num
    # print('rho_a', rho_a)

    # iteraton of the pa_dic
    t_list = []
    rho_list = []
    temp_rho = 0
    neigbor_dic_a = get_nei_dic(net_a)

    for t in range(max_step):
        # 修改后的计算A态自媒体节点比例逻辑（与MMCA和MC函数保持一致）
        # 计算A态自媒体节点的总概率
        fac_a_media = 0
        for node in media_nodes:
            fac_a_media += pa_dic[node]  # 直接累加概率值

        # 计算A态自媒体节点比例 ρA
        media_node_count = len(media_nodes)
        rho_a_media = fac_a_media / media_node_count if media_node_count > 0 else 0

        # Media environment perception probability (与其他两个函数逻辑一致)
        me = rho_a_media * m_0

        # print(me)

        # calculate the theta_dic
        theta_dic = {}
        for node in node_list:
            theta_i = 1
            for j in neigbor_dic_a[node]:
                theta_i = theta_i * (1 - (1 - rho_m) * pa_dic[j] * lamb - rho_m * pa_dic[
                    j] * lamb_1)  # the node in U state did not receive message from all the neighbors
            theta_dic[node] = theta_i
        new_pa_dic = {}
        rho_a = 0
        for i in node_list:
            m = me
            # print(m)
            pa = pa_dic[i]
            theta = theta_dic[i]
            new_pa = (1 - pa) * (1 - theta * (1 - m)) + pa * (1 - delta * (1 - m))
            new_pa_dic[i] = new_pa
            rho_a += new_pa
        # save time
        # print(pa_dic)
        rho_a = rho_a / node_num
        if (rho_a - temp_rho) < min_err and (t > min_step):
            break
        temp_rho = rho_a
        pa_dic = new_pa_dic.copy()
        rho_list += [rho_a]
        t_list += [t]
    # return t_list, rho_list

    # creat the matrix H
    neigbor_dic_b = get_nei_dic(net_b)
    h = np.zeros((node_num, node_num))
    for i in range(node_num):
        node_i = node_list[i]
        pa = pa_dic[node_i]
        for node_j in neigbor_dic_b[node_i]:
            j = node_list.index(node_j)
            h[j, i] = (1 + (gamma - 1) * pa)
    # calculate the largest real eigenvalues
    eigenvalue, featurevector = np.linalg.eig(h)
    eigen_list = np.real(eigenvalue)
    v_max = max(eigen_list)
    thred = mu / v_max
    # print(v_max)
    return thred

# monte-carlo simulation
def mc_of_uau_sis_mk_for_multi_process(net_a, net_b, initial_p, inform_p, lamb, beta_u, beta_a, delta,
                                       sigma, mu, rho_m, alpha, m_0, max_step, min_step, min_err):
    """
    修改后的蒙特卡洛仿真函数，在UAU层中引入自媒体个体
    按照MMCA理论函数的逻辑进行修改
    """
    node_list = list(net_a.nodes())
    node_num = len(node_list)

    lamb_1 = 1 - (1 - lamb) ** alpha  # 自媒体传播概率

    # 初始化自媒体节点,随机选择10%的节点作为自媒体
    media_num = int(node_num * rho_m)
    media_nodes = random.sample(node_list, media_num)

    # 创建自媒体节点标识字典
    is_media_node = {node: (node in media_nodes) for node in node_list}

    # initial seeds
    seed_list = []
    seed_num = int(node_num * initial_p)
    while len(seed_list) <= seed_num:
        seed_index = random.randint(0, node_num - 1)
        seed = node_list[seed_index]
        if seed not in seed_list:
            seed_list = seed_list + [seed]

    neigbor_dic_a = get_nei_dic(net_a)
    neigbor_dic_b = get_nei_dic(net_b)

    # initial the dicts
    u_dic = {}  # U状态节点
    a_dic = {}  # A状态节点
    i_dic = {}  # I状态节点
    s_dic = {}  # S状态节点

    # 初始化节点状态
    for node in node_list:
        u_dic[node] = 0
        if node in seed_list:
            i_dic[node] = 0  # 初始感染者处于UI状态
        else:
            s_dic[node] = 0  # 其他节点处于US状态

    fac_list_s, fac_list_i, fac_list_u, fac_list_a, time_list = [], [], [], [], []
    temp_i = 0
    temp_a = 0

    for t in range(max_step):
        # 计算当前A态自媒体节点比例 ρA
        fac_a_media = 0
        for node in media_nodes:
            if node in a_dic:
                fac_a_media += 1

        media_node_count = len(media_nodes)
        rho_a = fac_a_media / media_node_count if media_node_count > 0 else 0
        me = rho_a * m_0  # 媒体环境感知概率

        # print(me)

        # UAU过程开始 - 修改为包含自媒体逻辑
        new_a_dic = {}
        new_u_dic = {}

        # A状态节点向U状态邻居传播（包含自媒体逻辑）
        for i in a_dic:
            neighbor_list = neigbor_dic_a[i]
            for j in neighbor_list:
                if j in u_dic:
                    # 根据传播者是否为自媒体节点选择不同的传播概率
                    if is_media_node[i]:
                        trans_prob = lamb_1  # 自媒体传播概率
                    else:
                        trans_prob = lamb  # 普通传播概率

                    temp = random.random()
                    if temp < trans_prob:
                        new_a_dic[j] = t

        # A状态节点以概率delta转为U状态
        for i in list(a_dic.keys()):
            temp = random.random()
            if temp < delta:
                new_u_dic[i] = t

        # 更新UAU状态
        a_dic.update(new_a_dic)
        a_dic = del_dic_b_from_dic_a(a_dic, new_u_dic)
        u_dic = del_dic_b_from_dic_a(u_dic, new_a_dic)
        u_dic.update(new_u_dic)

        #自媒体传播过程
        new_a_dic = {}
        for i in u_dic:
            temp = random.random()
            if(temp < me):
                new_a_dic[i] = t
        # update the mass-media states of nodes
        u_dic = del_dic_b_from_dic_a(u_dic, new_a_dic)
        a_dic.update(new_a_dic)

        # SIOS过程开始
        new_i_dic = {}
        new_s_dic = {}

        # 疾病传播：I状态节点向S状态邻居传播
        for i in i_dic:
            neighbor_list = neigbor_dic_b[i]
            for j in neighbor_list:
                if j in s_dic:
                    if j in u_dic:
                        temp = random.random()
                        if temp < beta_u:
                            new_i_dic[j] = t
                    elif j in a_dic:
                        temp = random.random()
                        if temp < beta_a:
                            new_i_dic[j] = t

        # 康复过程：I状态节点以概率mu转为S状态
        for i in list(i_dic.keys()):
            temp = random.random()
            if temp < mu:
                new_s_dic[i] = t

        # 更新SIOS状态
        s_dic.update(new_s_dic)
        s_dic = del_dic_b_from_dic_a(s_dic, new_i_dic)
        i_dic.update(new_i_dic)
        i_dic = del_dic_b_from_dic_a(i_dic, new_s_dic)

        # 自我觉醒过程（需要考虑媒体环境感知）
        new_a_dic = {}
        for i in u_dic:
            if i in i_dic:
                temp = random.random()
                if temp < sigma:
                    new_a_dic[i] = t
        # update the self-awakening states of nodes
        a_dic.update(new_a_dic)
        u_dic = del_dic_b_from_dic_a(u_dic, new_a_dic)

        # 计算各状态比例
        fac_u = len(u_dic) / node_num
        fac_a = len(a_dic) / node_num
        fac_s = len(s_dic) / node_num
        fac_i = len(i_dic) / node_num

        fac_list_s.append(fac_s)
        fac_list_i.append(fac_i)
        fac_list_u.append(fac_u)
        fac_list_a.append(fac_a)
        time_list.append(t)

        # 收敛检查
        if (fac_i - temp_i) < min_err and (fac_a - temp_a) < min_err and (t > min_step):
            return (fac_a, fac_i)
        temp_i = fac_i
        temp_a = fac_a

    return (fac_a, fac_i)


def mc_of_uau_sis_mk_multi_process(pro_num=None, repeat_time=None, net_a=None, net_b=None, initial_p=None,
                                   inform_p=None, lamb=None, beta_u=None, beta_a=None, delta=None,
                                   sigma=None, mu=None, rho_m=None, alpha=None, m_0=None, max_step=None, min_step=None, min_err=None):
    po = Pool(processes=pro_num)
    result = []
    for i in range(repeat_time):
        result.append(po.apply_async(mc_of_uau_sis_mk_for_multi_process,
                                     args=(net_a, net_b, initial_p, inform_p, lamb, beta_u, beta_a, delta,
                                           sigma, mu, rho_m, alpha, m_0, max_step, min_step, min_err)))
    po.close()
    po.join()
    fac_a = 0
    fac_i = 0
    for res in result:
        fac_a += res.get()[0]
        fac_i += res.get()[1]
    fac_a = fac_a / repeat_time
    fac_i = fac_i / repeat_time
    return fac_a, fac_i

def mmca_of_uau_sis_mk_multi_process(pro_num=None, beta_list=None, net_a=None, net_b=None, initial_p=None,
                                     inform_p=None, lamb=None, gamma=None, delta=None,
                                     sigma=None, mu=None, rho_m=None, alpha=None, m_0=None, max_step=None, min_step=None, min_err=None):
    po = Pool(processes=pro_num)
    result = []
    for beta in beta_list:
        result.append(po.apply_async(mmca_of_uau_sis_mk_for_multi_process,
                                     args=(net_a, net_b, initial_p, inform_p, lamb, beta, beta * gamma, delta,
                                           sigma, mu, rho_m, alpha, m_0, max_step, min_step, min_err)))
    po.close()
    po.join()
    fac_list_a = []
    fac_list_i = []
    for res in result:
        fac_a = res.get()[0]
        fac_i = res.get()[1]
        fac_list_a.append(fac_a)
        fac_list_i.append(fac_i)
    return fac_list_a, fac_list_i