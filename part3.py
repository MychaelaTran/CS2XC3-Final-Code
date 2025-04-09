
import random
import math
import heapq
from Graph import Graph


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
    pred_matrix = [ [-1] * numNodes for _ in range(numNodes) ] #same thing here but -1 fpr predecsor (no predeccors)
    #O(v^2) to make each 

    #run single soure disjstra on every ndoe
    for u in range(numNodes):
        dist_from_u, pred_from_u = dijkstra(graph, u)
        
        #update matrices
        for v in range(numNodes):
            dist_matrix[u][v] = dist_from_u[v]
            pred_matrix[u][v] = pred_from_u[v]
    
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
#returns this when neg cycle found ([-1000000000], [-1000000000]), GRaph has negative cycle
def allPairsNegative(graph : Graph):
    #check v times and if on vth run we can relax, then there is a negatice cycle and no shortest path 
    numNodes = graph.number_of_nodes()
    dist_matrix = [[math.inf for _ in range(numNodes)] for _ in range(numNodes)] #initalize inf as distance in 2d matrix
    pred_matrix = [[i if i == j else -1 for j in range(numNodes)] for i in range(numNodes)] #same thing here but -1 fpr predecsor (no predeccors)
    #O(v^2) to make each 



    for startNode in range(numNodes):
        weightsIteration, predecessorsIteration = bellmanFord(graph, startNode)
        if (weightsIteration, predecessorsIteration) == ([-1000000000],[-1000000000]):
            print("Graph has negative cycle")
            return 
        for node in range(numNodes):
            dist_matrix[startNode][node] = weightsIteration[node]
            pred_matrix[startNode][node] = predecessorsIteration[node]
    return dist_matrix, pred_matrix



def bellmanFord(graph, startNode):
    numNodes = graph.number_of_nodes()
    dist = [math.inf] * numNodes
    dist[startNode] = 0
    predecessors = [-1] * numNodes

    #relax every edge v-1 times
    for i in range(numNodes -1):
        for currNode in range(numNodes):
            for neighbour in graph.connected_nodes(currNode):
                weight = graph.weight[(currNode, neighbour)]
                if dist[currNode] != math.inf and dist[currNode] + weight < dist[neighbour]:
                    dist[neighbour] = dist[currNode] + weight
                    predecessors[neighbour] = currNode

    # one more iteration to check for neg weight cycles
    for currNode in range(numNodes):
        for neighbour in graph.connected_nodes(currNode):
            weight = graph.weight[(currNode, neighbour)]
            if dist[currNode] != math.inf and dist[currNode] + weight < dist[neighbour]:
                return [-1000000000], [-1000000000]  # neg cycle founf

    predecessors[startNode] = startNode
    return dist, predecessors


#floyd warshall, directly from lab3: 
def floyd_warshall(G : Graph):
    n = G.number_of_nodes() #num of nodes
    #making the ditst matrix 
    d = [[float('inf')] * n for _ in range(n)]
    for i in range(n):
        #distnace from a node to itself set to zeor 
        d[i][i] = 0  

    #fill im graph edge weights 
    weights = G.get_weights()
    for (u, v), w in weights.items():
        d[u][v] = w

    #the floyd warshall algorithm itself 
    for k in range(n):
        for i in range(n):  # we are iteratinf over all pairs of nodes (i,j)
            for j in range(n):
                if d[i][j] > d[i][k] + d[k][j]:  #if path from i to j thru k is shorter then update d[i][j] with k
                    d[i][j] = d[i][k] + d[k][j]
    return d

# test = generate_random_graphPos(5, 19)
# test1 = generate_random_graphNeg(5,18)
# print("this is the adjacey matruix\n",test1.get_graph())
# print("these are the weights\n",test1.get_weights())
# slay = allPairsNegative(test1)
# print("asnwer", slay)


