import os
import copy
import random
import time
from itertools import chain, combinations

import matplotlib.pyplot as plt
import networkx as nx


####################
# Randomly generate a connected DAG
def generate_random_dag(n, edge_density):
    """
    Input:
        n: int - Number of nodes in the DAG.
        edge_density: float - Probability of edge creation between any two nodes.

    Output:
        graph: DiGraph - A directed acyclic graph (DAG) generated randomly.
    """

    graph = nx.DiGraph()
    nodes = ['v' + str(i) for i in range(1, n + 1)]
    for node in nodes:
        graph.add_node(node)
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < edge_density:
                graph.add_edge(nodes[i], nodes[j])
    return graph


# Check connected components in a DAG
def connect_components(graph):
    """
    Input:
        graph: DiGraph - A directed acyclic graph (DAG).

    Output:
        None (modifies the input graph by connecting components if needed).
    """

    components = list(nx.weakly_connected_components(graph))
    num_components = len(components)
    if num_components > 1:
        for i in range(num_components - 1):
            root1 = next(iter(components[i]))
            root2 = next(iter(components[i + 1]))
            graph.add_edge(root1, root2)


## Randomly generate a connected DAG
def Generate_DAG(n, edge_density):
    """
    Input:
        n: int - Number of nodes in the DAG.
        edge_density: float - Probability of edge creation between any two nodes.

    Output:
        graph: dict - A connected DAG represented as a dictionary.
    """

    random_dag = generate_random_dag(n, edge_density)
    connect_components(random_dag)
    graph = {node: {} for node in random_dag.nodes}
    for node in random_dag.nodes:
        neighbors = list(random_dag.neighbors(node))
        neighbors = [neighbor for neighbor in neighbors if neighbor != node]
        graph[node] = {neighbor: 'b' for neighbor in neighbors}
    return graph


#########################################
# Convert DAG format
def convert_dag(dag_graph):  # dictionary to list
    """
    Input:
        dag_graph: dict - A DAG represented as a dictionary.  DAG={1:{2:'b',3:'b'},2:{3:'b'}}

    Output:
        new_dag_graph: dict - A DAG represented as a  list. DAG={1:[2,3],2:[3]}
    """
    new_dag_graph = {node: {} for node in dag_graph.keys()}
    for node, neighbors in dag_graph.items():
        new_dag_graph[node] = list(neighbors.keys())
    return new_dag_graph


################
# Output the reversed graoh
def reverse_graph(graph):
    """
    Input:
        graph: dict - A DAG represented as a dictionary.

    Output:
        reversed_graph: dict - The reversed DAG, where all edges are reversed.
    """

    reversed_graph = {node: {} for node in graph}
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            if neighbor not in reversed_graph:
                reversed_graph[neighbor] = {}
            reversed_graph[neighbor][node] = graph[node][neighbor]
    return reversed_graph


# Plot
def plot_result(result, file_path, title=None):
    """
    Input:
        result: dict - A DAG represented as a dictionary.
        file_path: str - The path where the plot will be saved.
        title: str or None - The title of the plot.

    Output:
        None (saves a plot of the DAG to the specified file path).
    """
    G = nx.DiGraph()
    # Add nodes
    for node in result:
        G.add_node(node)
        # Add edges and annotation information

    edge_colors = []
    for node, neighbors in result.items():
        for neighbor, info in neighbors.items():
            G.add_edge(node, neighbor)
            edge_colors.append(info)
            # plot
    pos = nx.circular_layout(G)  # Use the circular_layout layout method
    # Create an image of size 8x6 inches with dpi set to 150
    plt.figure(figsize=(16, 12), dpi=150)
    nx.draw_networkx(G, pos, with_labels=True, node_color='lightblue',
                     node_size=2000, font_size=28, edge_color=edge_colors)

    if title:
        plt.title(title)
    else:
        plt.title("initial")

    plt.savefig(file_path)
    # plt.show()
    plt.close()


#########################
# Children set of a vertex
def ch(DAG, vertices, reverse_DAG=None):
    """
    Input:
        DAG: dict - A DAG represented as a dictionary.
        vertices: str - A vertex in the DAG.
        reverse_DAG: dict or None - The reversed DAG, if already computed.

    Output:
        set - The set of children of the given vertex.
    """
    return set(DAG[vertices].keys())


# Parent set of  a vertex
def pa(DAG, vertices, reverse_DAG=None):
    """
    Input:
        DAG: dict - A DAG represented as a dictionary.
        vertices: str - A vertex in the DAG.
        reverse_DAG: dict or None - The reversed DAG, if already computed.

    Output:
        set - The set of parents of the given vertex.
    """

    if reverse_DAG is None:
        reverse_DAG = reverse_graph(DAG)
    return set(reverse_DAG[vertices].keys())


# Spouse set of a vertex
def tch(DAG, vertices, reverse_DAG=None):
    """
    Input:
        DAG: dict - A DAG represented as a dictionary.
        vertices: str - A vertex in the DAG.
        reverse_DAG: dict or None - The reversed DAG, if already computed.

    Output:
        set - The set of spouses of the given vertex.
    """

    if reverse_DAG is None:
        reverse_DAG = reverse_graph(DAG)
    tch_value = set()
    for i in ch(DAG, vertices, reverse_DAG):
        tch_value = tch_value.union(pa(DAG, i, reverse_DAG))
    tch_value.discard(vertices)
    return tch_value


# Spouse set of vertices
def set_tch(DAG, nodes, reverse_DAG=None):
    """
    Input:
        DAG: dict - A DAG represented as a dictionary.
        nodes: list - A list of vertices in the DAG.
        reverse_DAG: dict or None - The reversed DAG, if already computed.

    Output:
        set - The set of spouses of the given vertices.
    """

    if reverse_DAG is None:
        reverse_DAG = reverse_graph(DAG)
    stack = list(nodes)
    tch_set = set()
    for i in stack:
        for j in ch(DAG, i, reverse_DAG):
            tch_set = tch_set | pa(DAG, j, reverse_DAG)
        tch_set.discard(i)
    return tch_set


# Children set of vertices
def set_ch(DAG, nodes, reverse_DAG=None):
    """
    Input:
        DAG: dict - A DAG represented as a dictionary.
        nodes: list - A list of vertices in the DAG.

    Output:
        set - The set of children of the given vertices.
    """

    stack = list(nodes)
    ch = set()
    for i in stack:
        ch = ch | set(DAG[i].keys())
    return ch


# Parent set of vertices
def set_pa(DAG, nodes, reverse_DAG=None):
    """
    Input:
        DAG: dict - A DAG represented as a dictionary.
        nodes: list - A list of vertices in the DAG.
        reverse_DAG: dict or None - The reversed DAG, if already computed.

    Output:
        set - The set of parents of the given vertices.
    """

    stack = list(nodes)
    pa = set()
    if reverse_DAG is None:
        reverse_DAG = reverse_graph(DAG)
    for i in stack:
        pa = pa | set(reverse_DAG[i].keys())
    return pa


# Neighbor set of a vertex
def ne(DAG, node, reverse_DAG=None):
    """
    Input:
        DAG: dict - A DAG represented as a dictionary.
        node: str - A vertex in the DAG.
        reverse_DAG: dict or None - The reversed DAG, if already computed.

    Output:
        set - The set of neighbors of the given vertex.
    """

    if reverse_DAG is None:
        reverse_DAG = reverse_graph(DAG)
    ne_ = set(DAG[node].keys()) | set(reverse_DAG[node].keys())
    return ne_


# Neighbor set of vertices
def set_ne(DAG, nodes, reverse_DAG=None):
    """
    Input:
        DAG: dict - A DAG represented as a dictionary.
        nodes: list - A list of vertices in the DAG.
        reverse_DAG: dict or None - The reversed DAG, if already computed.

    Output:
        set - The set of neighbors of the given vertices.
    """

    stack = list(nodes)
    if reverse_DAG is None:
        reverse_DAG = reverse_graph(DAG)
    set_ne_ = set()
    for i in stack:
        set_ne_ = set_ne_ | ne(DAG, i, reverse_DAG)
    return set_ne_


# Ancestor set of vertices
def find_ancestors(DAG, nodes, reverse_DAG=None):
    """
    Input:
        DAG: dict - A DAG represented as a dictionary.
        nodes: list - A list of vertices in the DAG.
        reverse_DAG: dict or None - The reversed DAG, if already computed.

    Output:
        set - The set of ancestors of the given vertices.
    """

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


# Markov blanket of  a vertex
def markov_blanket(DAG, vertices, reverse_DAG=None):
    """
    Input:
        DAG: dict - A DAG represented as a dictionary.
        nodes: list - A list of vertices in the DAG.

    Output:
        set - The set of markov blanket of a given vertex.
    """

    if reverse_DAG is None:
        reverse_DAG = reverse_graph(DAG)
    pa_ = pa(DAG, vertices, reverse_DAG)
    ch_ = ch(DAG, vertices, reverse_DAG)
    tch_ = tch(DAG, vertices, reverse_DAG)
    markov = ch_ | pa_ | tch_
    return markov


# Markov blanket of  vertices
def set_markov_blanket(DAG, nodes, reverse_DAG=None, re=False):
    """
    Input:
        DAG: dict - A DAG represented as a dictionary.
        nodes: list - A list of vertices in the DAG.

    Output:
        set - The set of markov blanket of  given vertices.
    """

    if reverse_DAG is None:
        reverse_DAG = reverse_graph(DAG)
    set_pa_ = set_pa(DAG, nodes, reverse_DAG)
    set_ch_ = set_ch(DAG, nodes, reverse_DAG)
    set_tch_ = set_tch(DAG, nodes, reverse_DAG)
    set_markov = set_ch_ | set_pa_ | set_tch_
    return (set_pa_, set_ch_, set_tch_, set_markov) if re else set_markov


# G1-G2
def dict_difference(dict1, dict2):
    """
    Calculate the difference between two dictionaries.
    
    Input:
    - dict1: The first dictionary.
    - dict2: The second dictionary to compare with the first.
    
    Output:
    - difference: A dictionary containing the elements in dict1 that are not in dict2 or that differ between dict1 and dict2.
    """

    difference = {node: {} for node in dict1.keys()}
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


# Check  c-removability
def is_cremoved(G, vertices, reverse_DAG=None):
    """
    Check if a vertex is c-removable from the DAG.
    
    Input:
    - G: The original Directed Acyclic Graph (DAG).
    - vertices: The vertex to check for c-removability.
    - reverse_DAG: (Optional) The reversed DAG. If not provided, it will be computed.
    
    Output:
    - Boolean value indicating whether the vertex is c-removable.
    """

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


# Check sequentially c-removability
def is_set_cremoved(G, M):
    """
    Check if a set of vertices is sequentially c-removable from the DAG.
    
    Input:
    - G: The original DAG.
    - M: A list of vertices to check for sequential c-removability.
    
    Output:
    - A tuple containing a boolean indicating if the entire set is c-removable and a list of c-removable vertices in sequence.
    """
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

# Moralization on a DAG
def moralize_dag(dag, dag_reverse=None):
    """
    Moralize a DAG, connecting all parents of each node.
    
    Input:
    - dag: The original DAG.
    - dag_reverse: (Optional) The reversed DAG. If not provided, it will be computed.
    
    Output:
    - moralized_dag: The moralized DAG.
    """

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


# Obtain a induced graph from a DAG
def sub_dag(DAG, H):
    """
    Obtain a subgraph induced by a set of vertices from the DAG.
    
    Input:
    - DAG: The original DAG.
    - H: A set of vertices to induce the subgraph.
    
    Output:
    - sub_DAG: The induced subgraph.
    """

    sub_DAG = dict()
    for i, j_dict in DAG.items():
        if i in H:
            sub_DAG[i] = dict()
            for j, value in j_dict.items():
                if j in H:
                    sub_DAG[i][j] = value
    return sub_DAG


# Determine if two vertices are connected in a UG
def is_connected(graph, start, target):
    """
    Determine if two vertices are connected in an undirected graph.
    
    Input:
    - graph: The undirected graph.
    - start: The starting vertex.
    - target: The target vertex.
    
    Output:
    - Boolean value indicating whether there is a path connecting the start and target vertices.
    """

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


# Check if there exists an inducing path between two vertices in a DAG
def is_inducing_path(DAG, u, v, M, reverse_DAG=None, re_MF=False):
    """
    Check if there exists an inducing path between two vertices in a DAG.
    
    Input:
    - DAG: The original DAG.
    - u: The first vertex.
    - v: The second vertex.
    - M: A set of vertices to consider for the inducing path.
    - reverse_DAG: (Optional) The reversed DAG. If not provided, it will be computed.
    - re_MF: (Optional) A boolean to determine if the function should return additional details.   
    Output:
    - Boolean value indicating whether there exists an inducing path between u and v.
      If re_MF is True, a tuple containing the vertices and the boolean value is returned.
      """

    if reverse_DAG is None:
        reverse_DAG = reverse_graph(DAG)
    nodes = [u, v]
    AN = find_ancestors(DAG, nodes, reverse_DAG)
    DAG_AN = sub_dag(DAG, list(AN))  # ancestral graph
    MDAG_AN = moralize_dag(DAG_AN)  # ancestral graph after moralizing
    MDAG_M_u_v = sub_dag(MDAG_AN, M + nodes)  # E
    flag = is_connected(MDAG_M_u_v, u, v)  # V+E
    return (nodes, flag) if re_MF else flag


# Check t-removability
def ITRSA1_(DAG, M, R1=None, reverse_DAG=None):
    """
    Check t-removability of a set of vertices in a DAG and return the flag and list of inducing paths.
    
    Input:
    - DAG: The original DAG.
    - M: The set of vertices to check for t-removability.
    - R1: (Optional) The set of vertices not in M. If not provided, it will be computed.
    - reverse_DAG: (Optional) The reversed DAG. If not provided, it will be computed.
    
    Output:
    - A tuple containing a boolean indicating t-removability and a list of mf-pairs .
    """

    if reverse_DAG is None:
        reverse_DAG = reverse_graph(DAG)
    if R1 is None:
        R1 = [x for x in DAG if x not in M]
    MF = list()
    f = True
    for i in range(1, len(R1)):
        for j in range(i):  #
            if R1[j] not in DAG[R1[i]] and R1[i] not in DAG[R1[j]]:
                nodes, flag = is_inducing_path(DAG, R1[i], R1[j], M, reverse_DAG, re_MF=True)
                if flag:
                    f = False
                    MF.append(nodes)
    return f, MF


# Check t-removability
def ITRSA_(DAG, M, R1=None, reverse_DAG=None):
    """
    Check t-removability of a set of vertices in a DAG.
    
    Input:
    - DAG: The original DAG.
    - M: The set of vertices to check for t-removability.
    - R1: (Optional) The set of vertices not in M. If not provided, it will be computed.
    - reverse_DAG: (Optional) The reversed DAG. If not provided, it will be computed.
    
    Output:
    - Boolean value indicating t-removability.
    """

    if reverse_DAG is None:
        reverse_DAG = reverse_graph(DAG)
    if R1 is None:
        R1 = [x for x in DAG if x not in M]
    for i in range(1, len(R1)):
        for j in range(i):  # V^2
            if R1[j] not in DAG[R1[i]] and R1[i] not in DAG[R1[j]]:
                if is_inducing_path(DAG, R1[i], R1[j], M, reverse_DAG):
                    return False
    return True


# Check t-removability
def ITRSA(DAG, M, re_MF=False):
    """
    Check the t-removability of a set of vertices.

    Input:
    - DAG: The original Directed Acyclic Graph (DAG).
    - M: The set of vertices to check for t-removability.
    - re_MF: (Optional) If True, returns detailed information about the mf-pairs; 
             otherwise, returns only a boolean value.

    Output:
    - If re_MF is True, returns a tuple with a boolean indicating t-removability and a list of mf-pairs;
      otherwise, returns a boolean indicating whether the set of vertices is t-removable.
    """

    R = [x for x in DAG if x not in M]
    reverse_DAG = reverse_graph(DAG)
    set_markov = set_markov_blanket(DAG, M, reverse_DAG)
    R1 = [node for node in R if node in set_markov]
    if re_MF:
        return ITRSA1_(DAG, M, R1, reverse_DAG)
    else:
        return ITRSA_(DAG, M, R1, reverse_DAG)


# Check if vertices of pa(w) are adjacent unless both of them belongs to R
def is_pd(G, Ma, pa1):
    """
    Check if the parent nodes of a vertex in the graph are adjacent, unless both belong to the set R.

    Input:
    - G: The original graph.
    - Ma: The set of vertices to check.
    - pa1: The set of parent nodes of Ma.

    Output:
    - Returns True if the parent nodes are adjacent or if both belong to R; otherwise, returns False.
    """

    Ma = list(Ma)
    pa1 = list(pa1)
    for i in range(1, len(Ma)):
        for j in range(i):
            if Ma[i] not in pa1 or Ma[j] not in pa1:
                if Ma[i] not in G[Ma[j]] and Ma[j] not in G[Ma[i]]:
                    return False
    return True


# Check condition of c-removability
def condition(DAG, M, R, reverse_DAG=None):
    """
    Check the condition for c-removability in a Directed Acyclic Graph (DAG).

    Input:
    - DAG: The original Directed Acyclic Graph (DAG).
    - M: The set of vertices to check for c-removability.
    - R: The set of remaining vertices after excluding M.
    - reverse_DAG: (Optional) The reversed DAG. If not provided, it will be computed.

    Output:
    - Returns True if the condition for c-removability is met; otherwise, returns False.
    """

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


# Check condition of c-removability
def ICRSA(DAG, M):
    R = [x for x in DAG if x not in M]
    reverse_DAG = reverse_graph(DAG)
    set_markov = set_markov_blanket(DAG, M, reverse_DAG)
    R1 = [node for node in R if node in set_markov]
    sub_DAG = sub_dag(DAG, R1 + M)
    Sub_reverse_DAG = reverse_graph(sub_DAG)
    return condition(sub_DAG, M, R1, Sub_reverse_DAG) and ITRSA_(DAG, M, R1, reverse_DAG)


#############
if __name__ == "__main__":
    G = {
        'r1': {},
        'r2': {'r1': 'b'},
        'r3': {'r2': 'b'},
        'r4': {'r3': 'b'},
        'm1': {'r1': 'b'},
        'm2': {'m1': 'b', 'm3': 'b'},
        'm3': {'r4': 'b'},
        'm4': {'m3': 'b', 'r5': 'b'},
        'r5': {}
    }
    plot_result(G, 'j.png')
    R = ['r1', 'r3', 'r2', 'r4', 'r5']
    M = ['m1', 'm2', 'm3', 'm4']
    print(ITRSA(G, M, re_MF=True))
    print(ITRSA1_(G, M))
