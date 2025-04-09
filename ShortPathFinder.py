from abc import ABC, abstractmethod
from typing import List, Dict 

from part3 import bellmanFord, dijkstra
from Graph import Graph as g
from part4 import a_star

class Graph(ABC):

    @abstractmethod
    def get_adj_nodes(self, node: int) -> List[int]:
        pass

    @abstractmethod
    def add_node(self, node: int):
        pass

    @abstractmethod
    def add_edge(self, start: int, end: int, weight: float):
        pass

    @abstractmethod
    def get_num_of_nodes(self) -> int:
        pass

    #mention in report how we had to rename this to allow the inherited graph to work
    def w1(self, node: int) -> float:
        return 0 

class WeightedGraph(Graph):
    def __init__(self, nodes):
        self.graph = g(nodes)

    def get_adj_nodes(self, node: int) -> List[int]:
        return self.graph.get_graph().get(node, [])

    def add_node(self, node: int):
        self.graph.add_node_explicit(node)

    def add_edge(self, start: int, end: int, weight: float):
        self.graph.add_edge(start, end, weight)

    def get_num_of_nodes(self) -> int:
        return self.graph.number_of_nodes()

    def w(self, node1: int, node2: int) -> float:
        return self.graph.get_weights()[node1][node2]

class HeuristicGraph(WeightedGraph):
    def __init__(self, heuristic: Dict[int, float]):
        super().__init__(len(heuristic)) # we assume the heuristic is complete
        self.heuristic = heuristic

    def get_heuristic(self) -> Dict[int, float]:
        return self.heuristic




class SPAlgorithm(ABC):

    @abstractmethod
    def calc_sp(self, graph: WeightedGraph, source: int, dest: int) -> float:
        pass

class Dijkstra(SPAlgorithm):
    def calc_sp(self, graph: WeightedGraph, source: int, dest: int) -> float:
        return dijkstra(graph.graph, source)[0][dest]


class Bellman_Ford(SPAlgorithm):
    def calc_sp(self, graph: WeightedGraph, source: int, dest: int) -> float:
        return bellmanFord(graph, source)[0][dest]

class A_Star(SPAlgorithm):
    def __init__(self, heuristic: HeuristicGraph):
        self.heuristic = heuristic

    def calc_sp(self, graph: WeightedGraph, source: int, dest: int) -> float:
        _, path = a_star(graph, source, dest, self.heuristic)
        
        if not path:
            return float('inf') 

        total_cost = 0
        weights = graph.graph.get_weights()
        for i in range(len(path) - 1):
            total_cost += weights[(path[i], path[i + 1])]
        
        return total_cost


class ShortPathFinder:
    def calc_short_path(self, source: int, dest: int) -> float:
        return self.algorithm.calc_sp(self.graph, source, dest)
    def set_graph(self, graph: WeightedGraph):
        self.graph = graph
    def set_algorithm(self, algorithm: SPAlgorithm):
        self.algorithm = algorithm



