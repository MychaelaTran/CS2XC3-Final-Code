import heapq
from part3 import Graph


def a_star(graph, source, destination, heuristic):
    # get info
    adjacency = graph.get_graph()
    weights = graph.get_weights()

    # initial g_cost.. set all to infinity except the source
    g_cost = {node: float("inf") for node in adjacency}
    g_cost[source] = 0

    # preceeding node for path reconstruction
    predecessor = {node: None for node in adjacency}

    # priority queue will store tuples of the form (f_cost, node)
    # where f_cost = g_cost(node) + heuristic(node)
    # this is our openSet
    priority_queue = []
    heapq.heappush(priority_queue, (heuristic[source], source))

    # To keep track of visited nodes (optimization so we don't revisit processed nodes)
    visited = set()

    while priority_queue:
        # get the node with smallest f_cost
        current_f, current_node = heapq.heappop(priority_queue)

        # if we've seen this node's best path before, skip
        if current_node in visited:
            continue
        visited.add(current_node)

        # if we reached the destination we can stop. we found the optimal path
        if current_node == destination:
            break

        # explore neighbors, this is basically copy paste from the lecture notes
        for neighbor in adjacency[current_node]:
            # cost so far + edge weight
            tentative_g = g_cost[current_node] + weights[(current_node, neighbor)]
            if tentative_g < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g
                # ensure we update the predecessors
                predecessor[neighbor] = current_node
                # f_cost = g_cost + heuristic
                f_cost = tentative_g + heuristic[neighbor]
                heapq.heappush(priority_queue, (f_cost, neighbor))

    # reconstruct the path if we reached the destination
    path = []
    if g_cost[destination] < float("inf"):
        node = destination
        while node is not None:
            path.append(node)
            node = predecessor[node]
        path.reverse()

    return predecessor, path
