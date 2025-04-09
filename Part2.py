import random
import math
import heapq
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
        if(dist <= distTo[this_node] and counter[this_node] < k):
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
    counter[node] += 1
    weights = g.get_weights()
    for w in g.get_graph()[node]:
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

    for i in range(k):
        for node in graph.get_graph():
            for w in graph.get_graph()[node]:
                if (distTo[w] > distTo[node] + weights[(node,w)]):
                    distTo[w] = distTo[node] + weights[(node,w)]
                    edgeTo[w] = node
    
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