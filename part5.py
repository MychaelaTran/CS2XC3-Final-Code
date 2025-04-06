import heapq
from part3 import Graph
import csv
from part4 import a_star
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import math
#a* doesnt make sense only desisnged single src single dest
#doesjt make sense other context bc if explore all nodes, then same as disjtra, it explores all paths
#just ends up adding everything to find all paths     

#each statuoin bas node in graph
#edge exist if two stations connect4ed
#weigts of edgfes, use lat and long for each station to find the distance traveleed between the two stations
#that distance can servce as teh weight fir a fiven edge
#heruristic use the physical direct distance, not driving distance betweem the srouce adn a given station 
#parameter is string name of csv station file name

#WHY IS THE PROF ,MISSING STATIPN 189 I HAD TO CHNAGE MY WHOLE CODE IM DEAD AND LOSING MY MIND
def read_stations():
    station_pos = {}
    with open("london_stations.csv", 'r') as file: 
        csvreader = csv.DictReader(file)
        next(csvreader) 
        for row in csvreader:
            station_id = int(row["id"])
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            station_pos[station_id] = (lat, lon)

    #i create a mappinf from the station ids to an indexed number in order 
    station_ids = sorted(station_pos.keys())
    id_to_index = {station_id: index for index, station_id in enumerate(station_ids)}

    #cfeate new station pos using our new indeices 
    good_station = {id_to_index[station_id]: pos for station_id, pos in station_pos.items()}
    return good_station, id_to_index



# slay1 = read_stations()
# print(slay1)


def read_connections(id_to_index):
    connected_stations = {}
    with open("london_connections.csv", 'r') as file: 
        csvreader = csv.DictReader(file)
        next(csvreader)
        for row in csvreader:
            st1 = int(row["station1"])
            st2 = int(row["station2"])
            time_val = int(row["time"])

            #need to mao the station to its index
            if st1 in id_to_index and st2 in id_to_index:
                connected_stations[(id_to_index[st1], id_to_index[st2])] = time_val
    return connected_stations


def euclidean_dist(station_a, station_b, station_pos):
    (lat1, lon1) = station_pos[station_a]
    (lat2, lon2) = station_pos[station_b]
    return math.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

#Therefore, you can create a hashmap or a function, which serves
#as a heuristic function for A*, takes the input as a given station and returns the distance between
#source and the given station. 
def build_heuristic_dict(station_pos, dest):
    #this dict is  station_id : euclidean distance from station to dest
    dest_lat, dest_lon = station_pos[dest]
    heuristic_dict = {}
    for st_id, (lat, lon) in station_pos.items():
        heuristic_dict[st_id] = math.sqrt((lat - dest_lat)**2 + (lon - dest_lon)**2)
    return heuristic_dict



def london_graph(station_pos, connected_stations):
    num_stations = len(station_pos)
    G = Graph(num_stations)
    
    for (st1, st2) in connected_stations:
        if st1 in station_pos and st2 in station_pos:
            #find eucidean distacne to get the edge weighth 
            dist = euclidean_dist(st1, st2, station_pos)
            G.add_edge(st1, st2, dist)

    return G



#test = london_graph(stations, connections)
#print("this is test", test.graph)
#print("this is wegith", test.weight)


#dijstra with end node and constructing path to node back import heapq
def dijkstra(graph, source, destination):
    adjacency = graph.get_graph()
    weights = graph.get_weights()

    num_nodes = graph.number_of_nodes()
    visited = set()

    dist = {node: float("inf") for node in adjacency}
    dist[source] = 0

    predecessor = {node: None for node in adjacency}

    min_pq = []
    heapq.heappush(min_pq, (0, source))

    while min_pq:
        curr_dist, node = heapq.heappop(min_pq)

        if node in visited:
            continue
        visited.add(node)

        #only diff thing here we break once found so match a*
        if node == destination:
            break

        for neighbor in adjacency[node]:
            weight = weights[(node, neighbor)]
            test_dist = curr_dist + weight
            if test_dist < dist[neighbor]:
                dist[neighbor] = test_dist
                predecessor[neighbor] = node
                heapq.heappush(min_pq, (test_dist, neighbor))

    #reconstruct path to node like a*
    path = []
    if dist[destination] < float("inf"):
        node = destination
        while node is not None:
            path.append(node)
            node = predecessor[node]
        path.reverse()

    return predecessor, path



def experiment(graph: Graph, station_pos):
    num_nodes = graph.number_of_nodes()

    #dicts are of the form where the key = (src, dest) and  value = time taken
    dijkstra_times = {}
    a_star_times = {}

    for src in range(num_nodes):
        #do hesuristic for all nodes one time 
        all_heuristics = { dest: build_heuristic_dict(station_pos, dest) for dest in range(num_nodes) }
        #all_heursitics[1][2] would be the euclid distance from node 2-1
        #since our build heusristic func makes a dictof a node as a key and its distance to dest
        #since fn = gn + hn since hn is cost from a node to our dest


        for dest in range(num_nodes):
            if src == dest:
                continue

            #dijstra times
            start_d = time.time()
            dijkstra(graph, src, dest) 
            end_d = time.time()
            total_time_d = end_d - start_d
            dijkstra_times[(src, dest)] = total_time_d

            #a star times 
            heuristic = all_heuristics[dest]
            start_a = time.time()
            a_star(graph, src, dest, heuristic)
            end_a = time.time()
            total_time_a = end_a - start_a
            a_star_times[(src, dest)] = total_time_a

    return dijkstra_times, a_star_times


def plot_all_pair_timings(dijkstra_times, a_star_times):
    #keep consisten order
    all_pairs = sorted(dijkstra_times.keys())

    dijkstra_vals = [dijkstra_times[pair] for pair in all_pairs]
    a_star_vals   = [a_star_times[pair] for pair in all_pairs]

    X_axis = np.arange(len(all_pairs))
    width = 0.4

    plt.figure(figsize=(20, 6)) 
    plt.bar(X_axis - width / 2, dijkstra_vals, width, label="Dijkstra", color="skyblue")
    plt.bar(X_axis + width / 2, a_star_vals, width, label="A*", color="salmon")

    #only show x ticks every 10,000th pair to reduce the messienss
    tick_step = max(len(X_axis) // 10, 1)
    tick_labels = [f"{src}->{dest}" for (src, dest) in all_pairs]
    plt.xticks(X_axis[::tick_step], tick_labels[::tick_step], rotation=45, fontsize=7)
    plt.xlabel("(src, dest) pair")
    plt.ylabel("time in seconds")
    plt.title("Dijkstra vs A* Timing for Each (src, dst) Pair")
    plt.legend()
    plt.tight_layout()
    plt.show()

def compare_algorithms_runtime(dijkstra_times, a_star_times):
    total_pairs = len(dijkstra_times)
    total_dijkstra_time = 0
    total_astar_time = 0
    dijkstra_faster_count = 0
    astar_faster_count = 0

    for pair in dijkstra_times:
        d_time = dijkstra_times[pair]
        a_time = a_star_times[pair]

        total_dijkstra_time += d_time
        total_astar_time += a_time

        #we ignore equal times
        if d_time < a_time:
            dijkstra_faster_count += 1
        elif a_time < d_time:
            astar_faster_count += 1
        

    avg_dijkstra_time = total_dijkstra_time / total_pairs
    avg_astar_time = total_astar_time / total_pairs

    print(f"avg dijk time: {avg_dijkstra_time:.9f} s")
    print(f"avg a star time:       {avg_astar_time:.9f} s\n")

    print(f"dijkstra was faster in :{dijkstra_faster_count} pairs")
    print(f"a start was faster in:  {astar_faster_count} pairs")

    return



station_pos, id_to_index = read_stations()
connections = read_connections(id_to_index)
graph = london_graph(station_pos, connections)

dijkstra_times, a_star_times = experiment(graph, station_pos)
print(dijkstra_times)
# compare_algorithms_runtime(dijkstra_times, a_star_times)
#plot_all_pair_timings(dijkstra_times, a_star_times)




#21-250
#78-202
#line path transfer data
def build_line_map(id_to_index):
    line_map = {}
    with open("london_connections.csv", "r") as file:
        reader = csv.DictReader(file)
        next(reader)
        for row in reader:
            s1 = int(row["station1"])
            s2 = int(row["station2"])
            line = int(row["line"])
            if s1 in id_to_index and s2 in id_to_index:
                u = id_to_index[s1]
                v = id_to_index[s2]
                line_map[(u, v)] = line
                line_map[(v, u)] = line  #undreictred line map since statin a-b same station b-a
    return line_map

def lines_used_in_path(path, line_map):
    used_lines = set()
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if (u, v) in line_map:
            used_lines.add(line_map[(u, v)])
    return used_lines

import random

def experiment_random_paths_with_lines(graph, station_pos, line_map, num_trials=30):
    num_nodes = graph.number_of_nodes()
    results = []

    trials_done = 0
    attempts = 0

    #can edit the num of trials
    while trials_done < num_trials:
        src = random.randint(0, num_nodes - 1)
        dest = random.randint(0, num_nodes - 1)
        if src == dest:
            continue

        #get the time for each path 
        _, path = dijkstra(graph, src, dest)
        #get the number of lines used/ transfers and which linse 
        lines_used = lines_used_in_path(path, line_map)
        #add all data to results as dicts per list item
        results.append({ "source": src, "destination": dest, "path": path, "lines_used": list(lines_used), "num_lines": len(lines_used) })
        trials_done += 1

    return results

#build the line map to see
line_map = build_line_map(id_to_index)
#print results 
results = experiment_random_paths_with_lines(graph, station_pos, line_map)
for iteration in results:
    print(f"{iteration['source']} → {iteration['destination']}, lines: {iteration['lines_used']}, transfers: {iteration['num_lines']}")





