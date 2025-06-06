import random
import math
import heapq
import time
import timeit 
from Graph import *



def dijkstra(graph: Graph, source, k):
    if k > (graph.number_of_nodes()-1) or k < 1:
        print("Invalid input!")
        return False

    distTo = {}
    for node in graph.get_graph():
        distTo[node] = float('inf')
    distTo[source] = 0

    edgeTo = {}

    for node in graph.get_graph():
        edgeTo[node] = None

    minheap = []

    heapq.heappush(minheap, (0, source))

    counter = {}

    for node in graph.get_graph():
        counter[node] = 0

    while minheap:
        dist, this_node = heapq.heappop(minheap)
        #skip if stale entuery 
        if dist > distTo[this_node]:
            continue
        relax(graph, this_node, minheap, distTo, edgeTo, counter, k)


    output = {}

    for node in graph.get_graph():
        path = [node]
        current = node
        while current != source and edgeTo[current] is not None:
            current = edgeTo[current]
            path.append(current)
        path = path[::-1]

        output[node] = (distTo[node], path)

    return output

def relax(g: Graph, node, minheap, distTo, edgeTo, counter, k):
    weights = g.get_weights()
    for w in g.get_graph()[node]:
        if(counter[w] < k):
            counter[w] += 1
            if (distTo[w] > distTo[node] + weights[(node,w)]):
                distTo[w] = distTo[node] + weights[(node,w)]
                edgeTo[w] = node
                heapq.heappush(minheap, (distTo[w], w))

def bellman_ford(graph: Graph, source, k):
    distTo = {}
    for node in graph.get_graph():
        distTo[node] = float('inf')
    distTo[source] = 0

    edgeTo = {}

    for node in graph.get_graph():
            edgeTo[node] = None

    weights = graph.get_weights()

    counter = {}

    for node in graph.get_graph():
        counter[node] = 0

    for i in range(graph.number_of_nodes() -1):
        for node in graph.get_graph():
            for w in graph.get_graph()[node]:
                if(counter[w] < k):
                    if (distTo[w] > distTo[node] + weights[(node,w)]):
                        distTo[w] = distTo[node] + weights[(node,w)]
                        edgeTo[w] = node
                    counter[w] += 1
    
    output = {}

    for node in graph.get_graph():
        path = [node]
        current = node
        while current != source and edgeTo[current] is not None:
            current = edgeTo[current]
            path.append(current)
        path = path[::-1]

        output[node] = (distTo[node], path)
    
    return output






#this function measures the accuracy of the shortest path given to the accurate shortest path 
def measure_accuracy(input, accurateOne):
    correct = 0
    total = len(input)
    for node in input: # check each node in inpt and compare distances for each node
        if input[node][0] == accurateOne[node][0]:
            correct += 1
    percentageAccuracy = (correct / total) * 100.0 #get it into percentage 
    return percentageAccuracy


#experiment where we run 200 teials on input node edges graph size
# and test different k values and compare the speed and accuracy of differnet k values 
def experiment(nodes, edges, trials=10):
    #we choose our k as fractions of (N-1)
    #cast to int so we don't deal with decimasl 
    k1 = max(1, int((nodes - 1) * 0.25))  # 25%
    k2 = max(1, int((nodes - 1) * 0.5))  # 50%
    k3 = max(1, int((nodes - 1) * 0.75))  # 75%
    k5 = max(1, int((nodes - 1) * 0.90))  # 90%
    k4 = nodes - 1                       # 100% 
    k_values = [k1, k2, k3, k5, k4]

    #use these to store our avg times and accuracy 
    dijkstra_times = {k: [] for k in k_values}
    dijkstra_acc   = {k: [] for k in k_values}
    bellman_times  = {k: [] for k in k_values}
    bellman_acc    = {k: [] for k in k_values}

    #our trials 
    for _ in range(trials):
        #make a new graph for each trial 
        G = generate_random_graphPos(nodes, edges)
        source = random.randint(0, nodes - 1)##pick randon node

        #Get the correct results (since k4 =100% so we know that this output will be correct and comparable for our others k)
        start_correct_d = time.time()
        accurate_dijkstra_output = dijkstra(G, source, k4)
        end_correct_d = time.time()
        start_accurate_b = time.time()
        accurate_bellman_output = bellman_ford(G, source, k4)
        end_accurate_b = time.time()


        #test each k 
        for k in [k1, k2, k3, k5]:
            #dijstrkar trials
            start_d = time.time()
            test_d = dijkstra(G, source, k) 
            end_d = time.time()
            dijkstra_times[k].append(end_d - start_d)
            dijkstra_acc[k].append(measure_accuracy(test_d, accurate_dijkstra_output))

            #bellamsn trials 
            start_b = time.time()
            partial_b = bellman_ford(G, source, k)
            end_b = time.time()
            bellman_times[k].append(end_b - start_b)
            bellman_acc[k].append(measure_accuracy(partial_b, accurate_bellman_output))

        
        dijkstra_times[k4].append(end_correct_d - start_correct_d)
        bellman_times[k4].append(end_accurate_b - start_accurate_b)
        dijkstra_acc[k4].append(100.0)
        bellman_acc[k4].append(100.0)

    #pront to terminal 
    print(f"RESULTS FOR GRAPH SIZE OF {nodes} NODES AND {edges} EDGES")
    for k in k_values:
        avg_d_time = sum(dijkstra_times[k])/len(dijkstra_times[k])
        avg_d_acc  = sum(dijkstra_acc[k])/len(dijkstra_acc[k])
        avg_b_time = sum(bellman_times[k])/len(bellman_times[k])
        avg_b_acc  = sum(bellman_acc[k])/len(bellman_acc[k])
        print(f"\nFor k={k}:")
        print(f"Dijkstra's average time: {avg_d_time:.5f} s, it's avg accuracy: {avg_d_acc:.2f}%")
        print(f"Bellman's average time: {avg_b_time:.5f} s, it's avg accuracy: {avg_b_acc:.2f}%")



experiment(500,5000)
