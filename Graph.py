import random
import math
import heapq
#weighted directed graph (took from my lab 3)
class Graph():
    #adjacney list 

    def __init__(self, nodes):
        self.graph = {}
        self.weight = {}
        for i in range(nodes):
            self.graph[i] = []

    def are_connected(self, node1, node2):
        for node in self.adj[node1]:
            if node == node2:
                return True
        return False

    def connected_nodes(self, node):
        return self.graph[node]

    def add_node(self,):
        #add a new node number = length of existing node
        self.graph[len(self.graph)] = []
    def add_node_explicit(self, node):
        self.graph[node] = []

    def add_edge(self, node1, node2, weight):
        if node2 not in self.graph[node1]:
            self.graph[node1].append(node2)
            self.weight[(node1, node2)] = weight



    def number_of_nodes(self,):
        return len(self.graph)

    def has_edge(self, src, dst):
        return dst in self.graph[src] 

    def get_weight(self,):
        total = 0
        for node1 in self.graph:
            for node2 in self.graph[node1]:
                total += self.weight[(node1, node2)]

        return total
    
    def get_graph(self,):
        return self.graph
    
    def get_weights(self,): 
        return self.weight




#extra functions to test our bellmna and dijkstra 
def generate_random_graphPos(nodes, edges):
    #makes random weighted graph thats underircted 
    G = Graph(nodes)


    edges_have = set()
    while len(edges_have) < edges:
        u, v = random.sample(range(nodes), 2)  #picks 2 distinct nodes
        weight = random.randint(1, 100)
        if (u, v) not in edges_have:
            G.add_edge(u, v, weight)
            edges_have.add((u, v))


    return G


def generate_random_graphNeg(nodes, edges):
    #makes random weighted graph thats underircted 
    G = Graph(nodes)


    edges_have = set()
    while len(edges_have) < edges:
        u, v = random.sample(range(nodes), 2)  #picks 2 distinct nodes
        weight = random.randint(-20, 80) #have negative numbers we can choose form, make it from -20,100 since I still want more postive weights than negative 
        if (u, v) not in edges_have:
            G.add_edge(u, v, weight)
            edges_have.add((u, v))


    return G
