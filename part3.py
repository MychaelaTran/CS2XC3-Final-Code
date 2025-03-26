
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

    def add_edge(self, node1, node2, weight):
        if node1 not in self.graph[node2]:
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


#uses a dijkstra approach with binary min heao
#returning 2dmatrix where the dist_matrix[i][j] is shortest patjh from i to j
#returning 2d matrix for predecessors where dist_matrix[i][j] is the predessor of j on shortest pathj from i to j (if -1 means no predeccors (ie all start nodes will have this))
#know dijstra is O((V+E)logV) - usually O(ELOGV) since edges dominate
#creating dist matrix are each V2 but thats smaller than O(VELOGV) for largegraphs insignificcant 
#if graph is dense than E = V2 and becomes O(V3LOGV)
#IF SPARSE then E = O(V) so then O(V2 LOGV)
#using heap is O(logn)
def allPairsPositive(graph : Graph) -> tuple[list[list[float]], list[list[int]]]: 
    numNodes = graph.number_of_nodes()
    dist_matrix = [[math.inf for _ in range(numNodes)] for _ in range(numNodes)] #initalize inf as distance in 2d matrix
    pred_matrix = [[i if i == j else -1 for j in range(numNodes)] for i in range(numNodes)] #same thing here but -1 fpr predecsor (no predeccors)
    #O(v^2) to make each 



    result = tuple[list[list[int]], list[list[int]]]
    for startNode in range(numNodes):
        weightsIteration, predecessorsIteration = dijkstra(graph, startNode)
        for node in range(numNodes):
            dist_matrix[startNode][node] = weightsIteration[node]
            pred_matrix[startNode][node] = predecessorsIteration[node]
    return dist_matrix, pred_matrix

def dijkstra(g  : Graph, startNode):
    numNodes = g.number_of_nodes()
    visited = set()
    predecessors = [-1] * numNodes
    dist =  [math.inf] * numNodes
    
    dist[startNode] = 0 #start node is 0 away from itself



    #create min pq that has tuples of (distacne, node) since heapq treats the firs telemtm of tuple as key for ordering 
    min_pq = []
    heapq.heappush(min_pq,(0,startNode))


    while min_pq:
        curr_dist, node = heapq.heappop(min_pq)

        if node not in visited: 
            visited.add(node)
            #relax all neighbours
            for neighbour in g.connected_nodes(node):
                if neighbour not in visited:
                    neighbourWeight = g.weight[(node, neighbour)]
                    test_dist = curr_dist + neighbourWeight
                    if test_dist < dist[neighbour]:
                        dist[neighbour] = test_dist
                        predecessors[neighbour] = node
                        heapq.heappush(min_pq, (test_dist, neighbour))
        else: 
            continue
    #add to make the paretn predecrrso itsefl for the node were at 
    predecessors[startNode] = startNode
    return dist, predecessors





#uses bellman ford approach
def allPairsNegativeBellman(graph : Graph):
    #check v times and if on vth run we can relax, then there is a negatice cycle and no shortest path 
    numNodes = graph.number_of_nodes()
    dist_matrix = [[math.inf for _ in range(numNodes)] for _ in range(numNodes)] #initalize inf as distance in 2d matrix
    pred_matrix = [[i if i == j else -1 for j in range(numNodes)] for i in range(numNodes)] #same thing here but -1 fpr predecsor (no predeccors)
    #O(v^2) to make each 



    for startNode in range(numNodes):
        weightsIteration, predecessorsIteration = bellmanFord(graph, startNode)
        print(weightsIteration)
        print(predecessorsIteration)
        if (weightsIteration, predecessorsIteration) == ([-1000000000],[-1000000000]):
            print("GRaph has negative cycle")
            return 
        for node in range(numNodes):
            dist_matrix[startNode][node] = weightsIteration[node]
            pred_matrix[startNode][node] = predecessorsIteration[node]
    return dist_matrix, pred_matrix



def bellmanFord(graph: Graph, startNode):
    numNodes = graph.number_of_nodes()
    dist = [math.inf] * numNodes
    dist[startNode] = 0
    predecessors = [-1] * numNodes

    #relax every edgfe v-1 times
    for i in range(numNodes + 1):
        for currNode in range(numNodes):
            for neighbour in graph.connected_nodes(currNode):
                weight = graph.weight[(currNode, neighbour)]
                if dist[currNode] != math.inf and dist[currNode] + weight < dist[neighbour]:
                    if i == numNodes:
                        print("neg ccle")
                        return [-1000000000],[-1000000000] #found neg cycle
                    else:
                        dist[neighbour] = dist[currNode] + weight
                        predecessors[neighbour] = currNode
    predecessors[startNode] = startNode


    return dist, predecessors




test = generate_random_graphPos(5, 19)
test1 = generate_random_graphNeg(5,12)
print("this is the adjacey matruix\n",test1.get_graph())
print("these are the weights\n",test1.get_weights())


#print("this is dijstra\n",dijkstra(test, 4))
print("this is bellman\n", bellmanFord(test1, 4))
#print("\nthis is all pairs postive",allPairsPositive(test))
print("\nthis is all pairs Bellamn",allPairsNegativeBellman(test1))





#the unknown function from lab 3 is the floys warshall algorthm 
#it handles computing all pairs shortest path and can deal with negative weights, we could also do this 
def allPairsNegativeFloyd(graph : Graph):
    
    #taking from lab 3
    def unknown(graph1 : Graph):
        n = graph1.number_of_nodes() #num of nodes
        for k in range(n):
            for i in range(n):  # we are iteratinf over all pairs of nodes (i,j)
                for j in range(n):
                    if graph1[i][j] > graph1[i][k] + graph1[k][j]:  #if path from i to j thru k is shorter then update d[i][j] with k
                        graph1[i][j] = graph1[i][k] + graph1[k][j]
        return graph1

    return unknown(graph)


#O(V3) time
#O(v2) space 
