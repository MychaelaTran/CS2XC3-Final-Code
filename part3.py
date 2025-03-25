
import random
#weighted undirected graph (took from my lab 3)
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

    def add_edge(self, node1, node2, weight):
        if node1 not in self.graph[node2]:
            self.graph[node1].append(node2)
            self.weight[(node1, node2)] = weight

            #since it is undirected
            #delete if directed
            self.graph[node2].append(node1)
            self.weight[(node2, node1)] = weight

    def number_of_nodes(self,):
        return len(self.graph)

    def has_edge(self, src, dst):
        return dst in self.graph[src] 

    def get_weight(self,):
        total = 0
        for node1 in self.graph:
            for node2 in self.graph[node1]:
                total += self.weight[(node1, node2)]
                
        # because it is undirected
        #delete if directed 
        return total/2

def generate_random_graph(nodes, edges):
    #makes random weighted graph thats underircted 
    G = Graph(nodes)


    edges_have = set()
    while len(edges_have) < edges:
        u, v = random.sample(range(nodes), 2)  #picks 2 distinct nodes
        weight = random.randint(1, 100)
        if (u, v) not in edges_have and (v, u) not in edges_have:
            G.add_edge(u, v, weight)
            edges_have.add((u, v))

    return G


#uses a dijkstra approach 
#returning matrix where the 1st matrix (row, col) is shortest path and 2nd is the node before
def allPairsPositive(graph : Graph) -> tuple[list[list[int]], list[list[int]]]: 
    for i in range(v):
        #run dijkstras
        #add results
        #TODO
        pass
    return

def dijkstra(graph):
    return 
    


#uses bellman ford approach
def allPairsNegative(graph):
    #check v times and if on vth run we can relax, then there is a negatice cycle and no shortest path 
    return 

#uses bellman ford approach
def allPairsNegative(graph):
    #check v times and if on vth run we can relax, then there is a negatice cycle and no shortest path 
    #TODO init distance matrix

    for i in range(v):
        for j in range(v):
            for k in range(v):
                relax(...)
    

#test