import os
import copy
import random
import time
from itertools import chain, combinations

import matplotlib.pyplot as plt
import networkx as nx


####################
# 随机生成一个连通的DAG
def generate_random_dag(n, edge_density):
    graph = nx.DiGraph()
    nodes = ['v' + str(i) for i in range(1, n + 1)]
    for node in nodes:
        graph.add_node(node)
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < edge_density:
                graph.add_edge(nodes[i], nodes[j])
    return graph


def connect_components(graph):
    components = list(nx.weakly_connected_components(graph))
    num_components = len(components)
    if num_components > 1:
        for i in range(num_components - 1):
            root1 = next(iter(components[i]))
            root2 = next(iter(components[i + 1]))
            graph.add_edge(root1, root2)


def Generate_DAG(n, edge_density):
    random_dag = generate_random_dag(n, edge_density)
    connect_components(random_dag)
    graph = {node: {} for node in random_dag.nodes}
    for node in random_dag.nodes:
        neighbors = list(random_dag.neighbors(node))
        neighbors = [neighbor for neighbor in neighbors if neighbor != node]
        graph[node] = {neighbor: 'b' for neighbor in neighbors}
    return graph


#########################################
# 转换DAG格式
def convert_dag(dag_graph):  # 字典到列表
    new_dag_graph = {node: {} for node in dag_graph.keys()}
    for node, neighbors in dag_graph.items():
        new_dag_graph[node] = list(neighbors.keys())
    return new_dag_graph


################

def reverse_graph(graph):
    reversed_graph = {node: {} for node in graph}
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            if neighbor not in reversed_graph:
                reversed_graph[neighbor] = {}
            reversed_graph[neighbor][node] = graph[node][neighbor]
    return reversed_graph


def plot_result(result, file_path, title=None):  # 画图
    G = nx.DiGraph()
    # 添加节点
    for node in result:
        G.add_node(node)
        # 添加边以及标注信息

    edge_colors = []
    for node, neighbors in result.items():
        for neighbor, info in neighbors.items():
            G.add_edge(node, neighbor)
            edge_colors.append(info)
            # 绘制图形
    pos = nx.circular_layout(G)  # 使用circular_layout布局方法    
    # 创建一个8x6英寸大小的图像，并将dpi设置为150
    plt.figure(figsize=(16, 12), dpi=150)
    nx.draw_networkx(G, pos, with_labels=True, node_color='lightblue',
                     node_size=2000, font_size=28, edge_color=edge_colors)

    if M:
        plt.title(title)
    else:
        plt.title("initial")

    plt.savefig(file_path)
    # plt.show()
    plt.close()


#########################
def ch(DAG, vertices, reverse_DAG=None):  # 顶点集的邻居
    return set(DAG[vertices].keys())


def pa(DAG, vertices, reverse_DAG=None):
    if reverse_DAG is None:
        reverse_DAG = reverse_graph(DAG)
    return set(reverse_DAG[vertices].keys())


def tch(DAG, vertices, reverse_DAG=None):
    if reverse_DAG is None:
        reverse_DAG = reverse_graph(DAG)
    tch_value = set()
    for i in ch(DAG, vertices, reverse_DAG):
        tch_value = tch_value.union(pa(DAG, i, reverse_DAG))
    tch_value.discard(vertices)
    return tch_value


def set_tch(DAG, nodes, reverse_DAG=None):
    if reverse_DAG is None:
        reverse_DAG = reverse_graph(DAG)
    stack = list(nodes)
    tch_set = set()
    for i in stack:
        for j in ch(DAG, i, reverse_DAG):
            tch_set = tch_set | pa(DAG, j, reverse_DAG)
        tch_set.discard(i)
    return tch_set


def set_ch(DAG, nodes, reverse_DAG=None):
    stack = list(nodes)
    ch = set()
    for i in stack:
        ch = ch | set(DAG[i].keys())
    return ch


def set_pa(DAG, nodes, reverse_DAG=None):
    stack = list(nodes)
    pa = set()
    if reverse_DAG is None:
        reverse_DAG = reverse_graph(DAG)
    for i in stack:
        pa = pa | set(reverse_DAG[i].keys())
    return pa


def ne(DAG, node, reverse_DAG=None):
    if reverse_DAG is None:
        reverse_DAG = reverse_graph(DAG)
    ne_ = set(DAG[node].keys()) | set(reverse_DAG[node].keys())
    return ne_


def ne(DAG, node, reverse_DAG=None):
    if reverse_DAG is None:
        reverse_DAG = reverse_graph(DAG)
    ne_ = set(DAG[node].keys()) | set(reverse_DAG[node].keys())
    return ne_


def set_ne(DAG, nodes, reverse_DAG=None):
    stack = list(nodes)
    if reverse_DAG is None:
        reverse_DAG = reverse_graph(DAG)
    set_ne_ = set()
    for i in stack:
        set_ne_ = set_ne_ | ne(DAG, i, reverse_DAG)
    return set_ne_


def find_ancestors(DAG, nodes, reverse_DAG=None):  # 顶点的祖先集
    if reverse_DAG is None:
        reverse_DAG = reverse_graph(DAG)
    visited = set()
    stack = list(nodes)
    ancestors = set()
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            ancestors.add(node)
            stack.extend(reverse_DAG.get(node, {}).keys())
    return ancestors


def markov_blanket(DAG, vertices, reverse_DAG=None):
    if reverse_DAG is None:
        reverse_DAG = reverse_graph(DAG)
    pa_ = pa(DAG, vertices, reverse_DAG)
    ch_ = ch(DAG, vertices, reverse_DAG)
    tch_ = tch(DAG, vertices, reverse_DAG)
    markov = ch_ | pa_ | tch_
    return markov


def set_markov_blanket(DAG, nodes, reverse_DAG=None, re=False):
    if reverse_DAG is None:
        reverse_DAG = reverse_graph(DAG)
    set_pa_ = set_pa(DAG, nodes, reverse_DAG)
    set_ch_ = set_ch(DAG, nodes, reverse_DAG)
    set_tch_ = set_tch(DAG, nodes, reverse_DAG)
    set_markov = set_ch_ | set_pa_ | set_tch_
    return (set_pa_, set_ch_, set_tch_, set_markov) if re else set_markov


def dict_difference(dict1, dict2):  # 两个图g1-g2
    difference = {node: {} for node in dict1.keys()}  # 创建一个包含所有节点的空字典
    for key in dict1:
        if key not in dict2:
            difference[key] = dict1[key]
        else:
            sub_dict = {}
            for sub_key in dict1[key]:
                if sub_key not in dict2[key] or dict1[key][sub_key] != dict2[key][sub_key]:
                    sub_dict[sub_key] = dict1[key][sub_key]
            if sub_dict:
                difference[key] = sub_dict
    return difference


##############
def is_cremoved(G, vertices, reverse_DAG=None):
    if reverse_DAG is None:
        reverse_DAG = reverse_graph(G)
    Ma = list(markov_blanket(G, vertices, reverse_DAG)) + [vertices]
    pa1 = list(pa(G, vertices, reverse_DAG))
    for i in range(1, len(Ma)):
        for j in range(i):
            if Ma[i] not in pa1 or Ma[j] not in pa1:
                if Ma[i] not in G[Ma[j]].keys() and Ma[j] not in G[Ma[i]].keys():
                    return False
    return True


def is_set_cremoved(G, M):
    G1 = copy.deepcopy(G)
    G2 = reverse_graph(G1)
    M1 = copy.deepcopy(M)
    k = 1
    M_T = []
    while M1:
        del1 = {}
        for i in M1:
            if is_cremoved(G1, i, G2):
                M_T.append(i)
                M1.remove(i)
                del1[i] = G1[i]
                for j in G2[i].keys():
                    if j not in del1:
                        del1[j] = {}
                    del1[j][i] = G2[i][j]
                G1 = dict_difference(G1, del1)
                G2 = dict_difference(G2, reverse_graph(del1))
                break
        if k != len(M_T):
            break
        else:
            k = k + 1
    return len(M_T) == len(M), M_T


###################

# 道德化 DAG  V*d^2
def moralize_dag(dag, dag_reverse=None):
    if dag_reverse is None:
        dag_reverse = reverse_graph(dag)
    dag1 = {}
    for node in dag_reverse:
        parents = list(dag_reverse[node].keys())
        for i in range(1, len(parents)):
            for j in range(i):
                parents_i = parents[i]
                parents_j = parents[j]
                if parents_i not in dag[parents_j] and parents_j not in dag[parents_i]:
                    if parents_i not in dag1:
                        dag1[parents_i] = {}
                    if parents_j not in dag1:
                        dag1[parents_j] = {}
                    dag1[parents_i][parents_j] = 'b'
                    dag1[parents_j][parents_i] = 'b'
    moralized_dag = {}
    graphs_to_merge = [dag, dag1, dag_reverse]
    for graph in graphs_to_merge:
        for node, neighbors in graph.items():
            moralized_dag.setdefault(node, {}).update(neighbors)
    return moralized_dag


################
def sub_dag(DAG, H):
    sub_DAG = dict()
    for i, j_dict in DAG.items():
        if i in H:
            sub_DAG[i] = dict()
            for j, value in j_dict.items():
                if j in H:
                    sub_DAG[i][j] = value
    return sub_DAG


def is_connected(graph, start, target):  # 判断两个点是否连接 V + E
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        visited.add(node)
        if node == target:
            return True
        for neighbor in graph.get(node, {}):
            if neighbor not in visited:
                stack.append(neighbor)
    return False


def is_inducing_path(DAG, u, v, M, reverse_DAG=None, re_MF=False):
    if reverse_DAG is None:
        reverse_DAG = reverse_graph(DAG)
    nodes = [u, v]
    AN = find_ancestors(DAG, nodes, reverse_DAG)  # 顶点的祖先集 E
    DAG_AN = sub_dag(DAG, list(AN))  # 祖先图  E
    MDAG_AN = moralize_dag(DAG_AN)  # 祖先图道德化 V*d^2
    MDAG_M_u_v = sub_dag(MDAG_AN, M + nodes)  # E
    flag = is_connected(MDAG_M_u_v, u, v)  # V+E
    return (nodes, flag) if re_MF else flag

def ITRSA1_(DAG, M, R1=None, reverse_DAG=None):
    if reverse_DAG is None:
        reverse_DAG = reverse_graph(DAG)
    if R1 is None:
        R1 = [x for x in DAG if x not in M]
    MF = list()
    f = True
    for i in range(1, len(R1)):
        for j in range(i):       #
            if R1[j] not in DAG[R1[i]] and R1[i] not in DAG[R1[j]]:
                nodes, flag = is_inducing_path(DAG, R1[i], R1[j], M, reverse_DAG, re_MF=True)
                if flag:
                    f = False
                    MF.append(nodes)
    return f, MF


def ITRSA_(DAG, M, R1=None, reverse_DAG=None):
    if reverse_DAG is None:
        reverse_DAG = reverse_graph(DAG)
    if R1 is None:
        R1 = [x for x in DAG if x not in M]
    for i in range(1, len(R1)):
        for j in range(i):       # V^2
            if R1[j] not in DAG[R1[i]] and R1[i] not in DAG[R1[j]]:
                if is_inducing_path(DAG, R1[i], R1[j], M, reverse_DAG):
                    return False
    return True


def ITRSA(DAG, M, re_MF=False):
    R = [x for x in DAG if x not in M]
    reverse_DAG = reverse_graph(DAG)
    set_markov = set_markov_blanket(DAG, M, reverse_DAG)
    R1 = [node for node in R if node in set_markov]
    # sub_DAG = sub_dag(DAG, R1 + M)
    # Sub_reverse_DAG = reverse_graph(sub_DAG)
    if re_MF:
        f, MF = ITRSA1_(DAG, M, R1, reverse_DAG)
    else:
        f = ITRSA_(DAG, M, R1, reverse_DAG)
    return (f, MF) if re_MF else f

def is_pd(G, Ma, pa1):
    Ma = list(Ma)
    pa1 = list(pa1)
    for i in range(1, len(Ma)):
        for j in range(i): # d^2
            if Ma[i] not in pa1 or Ma[j] not in pa1:
                if Ma[i] not in G[Ma[j]] and Ma[j] not in G[Ma[i]]:
                    return False
    return True


def condition(DAG, M, R, reverse_DAG=None):
    if reverse_DAG is None:
        reverse_DAG = reverse_graph(DAG)
    AN = find_ancestors(DAG, R, reverse_DAG)
    CH = set_ch(DAG, M, reverse_DAG) | set(M)
    w = AN & CH
    for i in w:  # V
        paa = pa(DAG, i, reverse_DAG)
        if not is_pd(DAG, paa, paa & set(R)):
            return False
    return True


def ICRSA(DAG, M):  # c可去集
    R = [x for x in DAG if x not in M]
    reverse_DAG = reverse_graph(DAG)
    set_markov = set_markov_blanket(DAG, M, reverse_DAG)
    R1 = [node for node in R if node in set_markov]
    sub_DAG = sub_dag(DAG, R1 + M)
    Sub_reverse_DAG = reverse_graph(sub_DAG)
    return condition(sub_DAG, M, R1, Sub_reverse_DAG) and ITRSA_(DAG, M, R1, reverse_DAG)


def powerset(vertices):  # 生成幂集
    # chain函数可以将多个迭代器连接在一起
    # combinations函数可以获取列表的所有可能组合
    return list(chain.from_iterable(combinations(vertices, r) for r in range(len(vertices) + 1)))


##########################
if __name__ == "__main__":
 
    
    G={
    'r1':{'m1':'a','m2':'b'},
    'r2':{'m1':'c','m2':'d','m3':'e','r3':'j'},
    'r3':{'m2':'f'},
    'm1':{'m3':'g'},
    'm2':{'m3':'h'},
    'm3':{}
   }
    R=['r1','r2','r3' ]
    M=['m1','m2','m3' ]
    print(ICRSA(G, M))
    


    
    G={
    'r1':{'u':'a','r2':'b'},
    'r3':{'v':'c'},
    'r2':{'r3':'g','v':'f'},
    'u':{'r2':'g','v':'e'},
    'v':{}
   }
    R=['r1','r2','r3' ]
    M=['u','v']
    print(ICRSA(G, M))
    
    G={
    'r1':{'m1':'a','m2':'b','r3':'c'},
    'r2':{'m1':'d','m2':'e','m3':'f','r3':'g'},
    'r3':{},
    'm1':{'m3':'h','r3':'i'},
    'm2':{'r3':'j'},
    'm3':{},
    'm4':{'m2':'k','r2':'l '},
   }
    R=['r1','r2','r3']
    M=['m1','m3','m2','m4']
    print(ITRSA_(G, M, R))
    print(is_set_cremoved(G, M))
    
    G = {
    'r1': {},
    'r2': {'r1': 'a'},
    'r3': {'r2': 'g'},
    'r4': {'r3': 'h'},
    'm1': {'r1': 'b'},
    'm2': {'m1': 'c', 'm3': 'd'},
    'm3': {'r4': 'g'},
    'm4': {'m3': 'e', 'r5': 'h'},
    'r5': {}
}
R = ['r1', 'r3', 'r2', 'r4', 'r5']
M = ['m1', 'm2', 'm3', 'm4']
print(ITRSA(G, M, re_MF=True))
print(ITRSA1_(G, M))
    # G = {
    #     'r1': {'m1': 'a', 'm2': 'b', 'r3': 'c'},
    #     'r2': {'m1': 'd', 'm2': 'e', 'r3': 'g', 'm3': 'b'},
    #     'r3': {},
    #     'm1': {'r3': 'i', 'm3': 'b'},
    #     'm2': {'r3': 'j'},
    #     'm3': {},
    #     'm4': {'m2': 'k', 'r2': 'l'},
    # }
    #
    # M = ['m3']
    # print(ICRSA(G, M))
    # n = 8  # 验证是否正确
    # density = 0.2  # 随机图的密度
    # c = 10
    # p = 0.4
    # print(f'DAG顶点个数: {n}, 边密度: {density}, M的比例：{p}')
    # DAG = Generate_DAG(n, density)  # 生成随机图
    # nodes = list(DAG.keys())
    #
    # M = random.sample(nodes, int(p * len(nodes)))
    #
    # M = list(M)
    # f, MF = ITRSA(DAG, M,True)
    # print(f'f={f}')
    # print(f'MF={MF}')
    # plot_result(DAG, 'ff.png', f'MF={MF},M={M}')
    # n = 10
    # density = 0.2 # 随机图的密度
    # p = 0.2  # M的比例
    # print(f'DAG顶点个数: {n}, 边密度: {density}, M的比例：{p}')
    # DAG = Generate_DAG(n, density)  # 生成随机图
    # nodes = list(DAG.keys())
    # M = random.sample(nodes, int(p * len(nodes)))
    # c = 5  # 重复实验的次数
    # start_time_DVE = time.time()
    # for i in range(c):
    #     f1, M_T = is_set_cremoved(DAG, M)
    # # 计算并打印执行时间
    # execution_time_DVE = time.time() - start_time_DVE
    # print(f"is_set_cremoved方法代码执行次数：{c}, 总时间: {execution_time_DVE}秒, 平均时间: {execution_time_DVE / c}秒")
    #
    # start_time_MDA = time.time()
    # for i in range(c):
    #     f2 = ICRSA(DAG, M)
    # # 计算并打印执行时间
    # execution_time_MDA = time.time() - start_time_DVE
    # print(f"ICRSA(DAG, M)方法代码执行次数：{c}, 总时间: {execution_time_MDA}秒, 平均时间: {execution_time_MDA / c}秒")
    # if f1 != f2:
    #     print('Flase')
    #     print(f'DAG={DAG}, M={M}')

    # DAG = {'alcoholism': {'THepatitis': 'a', 'Steatosis': 'b'},
    #      'vh_amn': {'ChHepatitis': 'c', 'hbsag': 'd', 'hbsag_anti': 'e', 'hbc_anti': 'f', 'hcv_anti': 'g',
    #                 'hbeag': 'h'},
    #      'hepatotoxic': {'THepatitis': 'i', 'RHepatitis': 'j'},
    #      'THepatitis': {'fatigue': 'k', 'phosphatase': 'l', 'inr': 'm', 'hepatomegaly': 'n', 'alt': 'o', 'ast': 'p',
    #                     'ggtp': 'q', 'anorexia': 'r', 'nausea': 's', 'spleen': 't'},
    #      'hospital': {'injections': 'u', 'transfusion': 'v'},
    #      'surgery': {'injections': 'w', 'transfusion': 'x'},
    #      'gallstones': {'choledocholithotomy': 'y', 'bilirubin': 'z', 'upper_pain': 'aa', 'fat': 'ab',
    #                     'pressure_ruq': 'ac', 'flatulence': 'ad', 'amylase': 'ae'},
    #      'choledocholithotomy': {'injections': 'af', 'transfusion': 'ag'},
    #      'injections': {'ChHepatitis': 'dr'},
    #      'transfusion': {'ChHepatitis': 'ds'},
    #      'ChHepatitis': {'fibrosis': 'ah', 'fatigue': 'ai', 'bilirubin': 'aj', 'pressure_ruq': 'ak',
    #                      'phosphatase': 'al', 'inr': 'am', 'ESR': 'an', 'alt': 'ao', 'ast': 'ap', 'ggtp': 'aq',
    #                      'cholesterol': 'ar', 'hbsag': 'as', 'hbsag_anti': 'at', 'hbc_anti': 'au', 'hcv_anti': 'av',
    #                      'hbeag': 'aw'},
    #      'sex': {'PBC': 'ax', 'Hyperbilirubinemia': 'ay'},
    #      'age': {'PBC': 'az', 'Hyperbilirubinemia': 'ba'},
    #      'PBC': {'bilirubin': 'bb', 'pressure_ruq': 'bc', 'ama': 'bd', 'le_cells': 'be', 'joints': 'bf', 'pain': 'bg',
    #              'platelet': 'bh', 'encephalopathy': 'bi', 'ESR': 'bj', 'ggtp': 'bk', 'cholesterol': 'bl',
    #              'carcinoma': 'bm'},
    #      'fibrosis': {'Cirrhosis': 'bn'},
    #      'diabetes': {'obesity': 'bo'},
    #      'obesity': {'Steatosis': 'bp'},
    #      'Steatosis': {'Cirrhosis': 'bq', 'triglycerides': 'br', 'pain_ruq': 'bs', 'hepatomegaly': 'bt', 'ESR': 'bu',
    #                    'alt': 'bv', 'ast': 'bw', 'ggtp': 'bx', 'cholesterol': 'by'},
    #      'Cirrhosis': {'bilirubin': 'bz', 'phosphatase': 'ca', 'proteins': 'cb', 'edema': 'cc', 'platelet': 'cd',
    #                    'inr': 'ce', 'alcohol': 'cf', 'encephalopathy': 'cg', 'alt': 'ch', 'ast': 'ci', 'spleen': 'cj',
    #                    'spiders': 'ck', 'albumin': 'cl', 'edge': 'cm', 'irregular_liver': 'cn', 'palms': 'co',
    #                    'carcinoma': 'cp'},
    #      'Hyperbilirubinemia': {'bilirubin': 'cq', 'pain_ruq': 'cr', 'inr': 'cs', 'hepatomegaly': 'ct', 'ESR': 'cu',
    #                             'ggtp': 'cv'},
    #      'RHepatitis': {'fatigue': 'cw', 'phosphatase': 'cx', 'hepatomegaly': 'cy', 'alt': 'cz', 'ast': 'da',
    #                     'ggtp': 'db', 'anorexia': 'dc', 'nausea': 'dd', 'spleen': 'de'},
    #      'bilirubin': {'itching': 'df', 'skin': 'dg', 'jaundice': 'dh'},
    #      'joints': {'pain': 'di'},
    #      'proteins': {'ascites': 'dj'},
    #      'platelet': {'bleeding': 'dk'},
    #      'inr': {'bleeding': 'dl'},
    #      'encephalopathy': {'urea': 'dm', 'density': 'dn', 'consciousness': 'do'},
    #      'hepatomegaly': {'hepatalgia': 'dp'},
    #      'hbsag': {'hbsag_anti': 'dq'},
    #      'triglycerides': {},
    #      'fatigue': {},
    #      'itching': {},
    #      'upper_pain': {},
    #      'fat': {},
    #      'pain_ruq': {},
    #      'pressure_ruq': {},
    #      'phosphatase': {},
    #      'skin': {},
    #      'ama': {},
    #      'le_cells': {},
    #      'pain': {},
    #      'edema': {},
    #      'bleeding': {},
    #      'flatulence': {},
    #      'alcohol': {},
    #      'urea': {},
    #      'ascites': {},
    #      'hepatalgia': {},
    #      'density': {},
    #      'ESR': {},
    #      'alt': {},
    #      'ast': {},
    #      'amylase': {},
    #      'ggtp': {},
    #      'cholesterol': {},
    #      'hbsag_anti': {},
    #      'anorexia': {},
    #      'nausea': {},
    #      'spleen': {},
    #      'consciousness': {},
    #      'spiders': {},
    #      'jaundice': {},
    #      'albumin': {},
    #      'edge': {},
    #      'irregular_liver': {},
    #      'hbc_anti': {},
    #      'hcv_anti': {},
    #      'palms': {},
    #      'hbeag': {},
    #      'carcinoma': {}
    #      }
    #
    # M = ["alcoholism", "vh_amn", "hepatotoxic", "THepatitis", "hospital",
    #      "surgery", "gallstones", "choledocholithotomy", "injections",
    #      "transfusion", "ChHepatitis", "fibrosis",
    #      "diabetes", "obesity", "Steatosis", "Cirrhosis",
    #      "triglycerides", "RHepatitis",
    #      "fatigue", "bilirubin", "itching", "upper_pain", "fat",
    #      "pain_ruq", "pressure_ruq", "phosphatase", "skin", "ama",
    #      "le_cells", "joints", "pain", "proteins", "edema", "platelet",
    #      "inr", "bleeding", "flatulence", "alcohol", "encephalopathy",
    #      "urea", "ascites", "hepatomegaly", "hepatalgia", "density",
    #      "ESR", "alt", "ast", "amylase", "ggtp", "cholesterol", "hbsag",
    #      "hbsag_anti", "anorexia", "nausea", "spleen", "consciousness",
    #      "spiders", "jaundice", "albumin", "edge", "irregular_liver",
    #      "hbc_anti", "hcv_anti", "palms", "hbeag", "carcinoma"]
    # c = 5 # 重复实验的次数
    # start_time_DVE = time.time()
    # for i in range(c):
    #     f1, M_T = is_set_cremoved(DAG, M)
    # # 计算并打印执行时间
    # execution_time_DVE = time.time() - start_time_DVE
    # print(f"is_set_cremoved方法代码执行次数：{c}, 总时间: {execution_time_DVE}秒, 平均时间: {execution_time_DVE / c}秒")
    #
    # start_time_MDA = time.time()
    # for i in range(c):
    #     f2 = ICRSA(DAG, M)
    # # 计算并打印执行时间
    # execution_time_MDA = time.time() - start_time_DVE
    # print(f"ICRSA(DAG, M)方法代码执行次数：{c}, 总时间: {execution_time_MDA}秒, 平均时间: {execution_time_MDA / c}秒")
    # if f1 != f2:
    #     print('Flase')
    #     print(f'DAG={DAG}, M={M}')

    ############################
    # n = 10  # 验证是否正确
    # density = 0.3  # 随机图的密度
    # c = 100
    # p = 0.2
    # print(f'DAG顶点个数: {n}, 边密度: {density}, M的比例：{p}')
    # DAG = Generate_DAG(n, density)  # 生成随机图
    # nodes = list(DAG.keys())
    #
    # M = random.sample(nodes, int(p * len(nodes)))
    #
    # M = list(M)
    # start_time_cremoved = time.time()
    # for i in range(c):
    #     f1, M_T = is_set_cremoved(DAG, M)
    # # 计算并打印执行时间
    # execution_time_cremoved = time.time() - start_time_cremoved
    # print(f'f1={f1},M={M}')
    # print(
    #     f"is_set_cremoved方法代码执行次数：{c}, 总时间: {execution_time_cremoved}秒, 平均时间: {execution_time_cremoved / c}秒")
    #
    # start_time_ICRSA = time.time()
    # for i in range(c):
    #     f2 = ICRSA(DAG, M)
    # # 计算并打印执行时间
    # execution_time_ICRSA = time.time() - start_time_ICRSA
    # print(
    #     f"ICRSA(DAG, M)方法代码执行次数：{c}, 总时间: {execution_time_ICRSA}秒, 平均时间: {execution_time_ICRSA / c}秒")
    # if f1 != f2:
    #     print('Flase')
    #     print(f'DAG={DAG}, M={M}')

    # ###############################
  