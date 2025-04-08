import heapq
from part3 import Graph
import csv
from part3 import dijkstra
from part4 import a_star
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from math import radians, cos, sin, asin, sqrt
#a* doesnt make sense only desisnged single src single dest
#doesjt make sense other context bc if explore all nodes, then same as disjtra, it explores all paths
#just ends up adding everything to find all paths     

#each statuoin bas node in graph
#edge exist if two stations connect4ed
#weigts of edgfes, use lat and long for each station to find the distance traveleed between the two stations
#that distance can servce as teh weight fir a fiven edge
#heruristic use the physical direct distance, not driving distance betweem the srouce adn a given station 
#parameter is string name of csv station file name

#WHY ARE WE MISSING STATIPN 189 I HAD TO CHNAGE MY WHOLE CODE IM DEAD AND LOSING MY MIND
def read_stations():
    station_pos = {}
    with open("london_stations.csv", 'r') as file: 
        csvreader = csv.DictReader(file)
        #skip the first line of the indicators
        next(csvreader) 
        for row in csvreader:
            station_id = int(row["id"])
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            station_pos[station_id] = (lat, lon)

    #i create a mappinf from the station ids to an indexed number in order 
    #use enumerate to create a dcit that maps each stations original id to a new contiguous index
    #do this bc the stattio  ids are not sequential, since 189 is missing 
    station_ids = sorted(station_pos.keys())
    id_to_index = {station_id: index for index, station_id in enumerate(station_ids)}

    #cfeate new station pos using our new indeices
    #to maje sure that when i iterate from 0 to the number of stations we cover all stsatuons even if the og ids were not seqentual 
    good_station = {id_to_index[station_id]: pos for station_id, pos in station_pos.items()}
    return good_station, id_to_index




def read_connections(id_to_index):
    connected_stations = {}
    with open("london_connections.csv", 'r') as file: 
        csvreader = csv.DictReader(file)
        next(csvreader)
        for row in csvreader:
            st1 = int(row["station1"])
            st2 = int(row["station2"])
            line = int(row["line"])  
            time_val = int(row["time"])

            #need to mao the station to its index
            if st1 in id_to_index and st2 in id_to_index:
                connected_stations[(id_to_index[st1], id_to_index[st2])] = line
    return connected_stations


#use haversine formula for lattitude longituude
def haversine(stationa, stationb, station_pos):
    (lat1, lon1) = station_pos[stationa]
    (lat2, lon2) = station_pos[stationb]

    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 
    return c * r
#returns the distance between the two points in kilometers



#ASSIGNMENT:: Therefore, you can create a hashmap or a function, which serves
#as a heuristic function for A*, takes the input as a given station and returns the distance between
#source and the given station. 
def build_heuristic_dict(station_pos, dest):
    #this dict is  station_id : euclidean distance from station to dest
    heuristic_dict = {}
    for st_id in station_pos:
        heuristic_dict[st_id] = haversine(st_id, dest, station_pos)
    return heuristic_dict



def london_graph(station_pos, connected_stations):
    num_stations = len(station_pos)
    G = Graph(num_stations)
    edge_lines = {}

    for (st1, st2), line in connected_stations.items():
        if st1 in station_pos and st2 in station_pos:
            dist = haversine(st1, st2, station_pos)
            G.add_edge(st1, st2, dist) 
            G.add_edge(st2, st1, dist) #undirected graph 
            #add the line this is on 
            edge_lines[(st1, st2)] = connected_stations[st1,st2]
            edge_lines[(st2, st1)] = connected_stations[st1,st2]

    return G, edge_lines



stations, index = read_stations()
connections =  read_connections(index)
# test, edges = london_graph(stations, connections)
# # print("this is test", test.graph)
# # print("this is wegith", test.weight)
# print("this is line connections",edges)


#dijstra with end node and constructing path to node back import heapq
def dijkstra2(graph, source, destination):
    #get graph and weights
    adjacency = graph.get_graph() 
    weights = graph.get_weights()
    num_nodes = graph.number_of_nodes()
    visited = set()
    #initzLie the distances to infintiy and the first nodeo to 0
    dist = {node: float("inf") for node in adjacency}
    dist[source] = 0
    #predecessor dict to track shortest path for later
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


#this experiemtn computes using the dijstra that has the src and destination and acts liek a* where dones one trialk per src,node pair
def experiment1(graph: Graph, station_pos):
    num_nodes = graph.number_of_nodes()

    #dicts are of the form where the key = (src, dest) and  value = time taken
    dijkstra_times = {}
    a_star_times = {}

    for src in range(num_nodes):
        #do hesuristic for all nodes one time 
        all_heuristics = {}
        for dest in range(num_nodes):
            heuristic = build_heuristic_dict(station_pos, dest)
            all_heuristics[dest] = heuristic
        #all_heursitics[1][2] would be the euclid distance from node 2-1
        #since our build heusristic func makes a dictof a node as a key and its distance to dest
        #since fn = gn + hn since hn is cost from a node to our dest


        for dest in range(num_nodes):
            if src == dest:
                continue

            #dijstra times
            start_d = time.time()
            dijkstra2(graph, src, dest) 
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

def run_experiment1():
    #get our data 
    station_pos, id_to_index = read_stations()
    connections = read_connections(id_to_index)
    #create oir graph, using our haversine
    graph, edges = london_graph(station_pos, connections)

    #run experiment1 that computes for all src dst pairs
    dijkstra_times, a_star_times = experiment1(graph, station_pos)
    
    #avg time over all src dest pairs 
    total_d_time = sum(dijkstra_times[x] for x in dijkstra_times)
    total_a_time = sum(a_star_times[x] for x in a_star_times)
    total_pairs = len(dijkstra_times) #just choose one bc dijstra and a star would be same length bc same trial 
    avg_d = total_d_time / total_pairs
    avg_a = total_a_time / total_pairs
    print(f"avg disktra time: {avg_d:.9f} s")
    print(f"avg a star time: {avg_a:.9f} s")

    #number of trials each was faster in
    dijkstra_faster = sum(1 for pair in dijkstra_times if dijkstra_times[pair] < a_star_times[pair])
    astar_faster = sum(1 for pair in dijkstra_times if a_star_times[pair] < dijkstra_times[pair])
    print(f"disjtra was faster in {dijkstra_faster} pairs")
    print(f"a start was faster in {astar_faster} pairs")

    #matplotlib
    d_times = [dijkstra_times[pair] for pair in sorted(dijkstra_times)]
    a_times = [a_star_times[pair] for pair in sorted(a_star_times)]
    x = np.arange(len(d_times))

    plt.figure(figsize=(14, 6))

    #plot the taller bar first, then the shorter bar over the talelr one so we can see both
    for i in range(len(x)):
        d = d_times[i]
        a = a_times[i]
        if d > a:
            plt.bar(x[i], d, color='skyblue', width=1)
            plt.bar(x[i], a, color='salmon', width=1)
        else:
            plt.bar(x[i], a, color='salmon', width=1)
            plt.bar(x[i], d, color='skyblue', width=1)

    plt.xticks([], [])
    plt.xlabel("Iterations")
    plt.ylabel("Time (seconds)")
    plt.title("Dijkstra vs A* for All Pairs -  (src,dst)")
    plt.tight_layout()
    plt.show()


# run_experiment1()


def experiment2(graph: Graph, station_pos):
    num_nodes = graph.number_of_nodes()
    #instead of 300x300 times, we only have 300, calclate time for single source shortest path to all ndoes, so run on each node 
    dijkstra_times = [None for _ in range(num_nodes)]
    a_star_times   = [None for _ in range(num_nodes)]
    #compute heuristic to use so dont have to make everytime
    all_heuristics = {dest: build_heuristic_dict(station_pos, dest) for dest in range(num_nodes) }

    for src in range(num_nodes):
        #disjtra has single run 
        #since dijstra alr computes sinlge src shortest path to all pther nodes, just calculate its time 
        start = time.time()
        dijkstra(graph, src)
        end = time.time()
        dijkstra_times[src] = end - start

        #a star needs one run per node from src
        #then add the time 
        total_astar_time = 0
        for dest in range(num_nodes):
            if src == dest:
                continue
            heuristic = all_heuristics[dest]
            start_a = time.time()
            a_star(graph, src, dest, heuristic)
            end_a = time.time()
            #add each time per src,dest
            total_astar_time += (end_a - start_a)
        #add final time for all src to dst per src
        a_star_times[src] = total_astar_time

    return dijkstra_times, a_star_times

 
def run_experiment2():
    #grab data
    station_pos, id_to_index = read_stations()
    connections = read_connections(id_to_index)
    graph, lines = london_graph(station_pos, connections)

    #run exp2
    dijkstra_times, a_star_times = experiment2(graph, station_pos)

    #averages
    num_nodes = len(dijkstra_times)
    avg_d = sum(dijkstra_times) / num_nodes
    avg_a = sum(a_star_times) / num_nodes
    print(f"avg disktra time: {avg_d:.9f} s")
    print(f"avg a star time: {avg_a:.9f} s")

    #count how many were faster 
    dijkstra_faster = sum(1 for i in range(num_nodes) if dijkstra_times[i] < a_star_times[i])
    astar_faster = sum(1 for i in range(num_nodes) if a_star_times[i] < dijkstra_times[i])
    print(f"disjtra was faster in {dijkstra_faster} srcs")
    print(f"a start was faster in {astar_faster} srcs")

    #MATPLOTLIB
    x = range(num_nodes)
    plt.figure(figsize=(12, 6))
    plt.plot(x, dijkstra_times, label="Dijkstra", color="skyblue")
    plt.plot(x, a_star_times, label="A*", color="salmon")
    plt.xlabel("Source Node Index")
    plt.ylabel("Total Time (seconds)")
    plt.title("Dijkstra vs A* For Total Time per Source Node")
    plt.legend()
    plt.tight_layout()
    plt.show()


#run_experiment2()





#LINE STUFF
#calculate how many lines/transfers we took for the shortest path 
#have dict of connections and their lines 
def build_line_map(connections):
    line_map = {}
    for (u, v), line in connections.items():
        line_map[(u, v)] = line 
        line_map[(v, u)] = line #make both ways 
    return line_map



#helper function to calcualte the number of lines used in shortest path 
def num_lines_used_in_path(path, line_map):
    used_lines = set()
    for i in range(len(path) - 1): #path could look lik [0,53,21,210], so check the lines used in every adjacent pairs
        x = path[i]
        y = path[i + 1] #look at luiesn used for each adjacent pari 
        used_lines.add(line_map[(x, y)])
    return used_lines


#tests 2 rnadom nodes and sees how many transfers/lines it uses and calculates the avg time for shortest path depdening on line treansfesrs 
#now I make an experiemnt to test if the number of trasnfers a shortste path has from src to dst impacts anything 
#use a* to test the imapct that line numbers have because I know a* is faster so decided to choose it
#the importnat part is i am keepng it constant that I am using a*
def random_node_test(graph, station_pos, connections, num_trials=200):
    num_nodes = graph.number_of_nodes()
    line_map = build_line_map(connections)  #build our line map to use to compute transfers later 
    same_line_times = []
    adjacent_line_times = []
    multi_transfer_times = []

    while num_trials > 0:
        #grab 2 ranbdom nodes to test
        src = random.randint(0, num_nodes - 1)
        dst = random.randint(0, num_nodes - 1)
        if src == dst: #skip if same station 
            continue

        heuristic = build_heuristic_dict(station_pos, dst)
        start = time.time() #time the a star algo with the 2 random nodes 
        pred, path = a_star(graph, src, dst, heuristic)
        end = time.time()
        time_taken = end - start

        lines_used = num_lines_used_in_path(path, line_map) #check which lines are used
        num_lines = len(lines_used) #count how many


        #match to rifht case 
        if num_lines == 1:
            same_line_times.append(time_taken)
        elif num_lines == 2:
            adjacent_line_times.append(time_taken)
        elif num_lines > 2:
            multi_transfer_times.append(time_taken)

        num_trials -= 1


    #averages
    same_line_avg = sum(same_line_times) / len(same_line_times) 
    adjacent_line_avg = sum(adjacent_line_times) / len(adjacent_line_times) 
    multi_transfers_avg = sum(multi_transfer_times) / len(multi_transfer_times) 
    print("same line avg: \n", same_line_avg)
    print("adj line avg: \n", adjacent_line_avg)
    print("milti tranfer avg: \n", multi_transfers_avg)
    

    #matplotlib
    avg_times = [same_line_avg, adjacent_line_avg, multi_transfers_avg]
    labels = ['Same Subway Line', 'Adjacent Subway Lines', 'Multiple Transfers']
    plt.figure(figsize=(8, 5))
    plt.bar(labels, avg_times, color=["pink", "skyblue", "green"])
    plt.ylabel("Average A* Time (µs)")
    plt.title("Line Transfers VS Shortest Path Run Time")
    plt.tight_layout()
    plt.show()
    return


stations, index = read_stations()
connections = read_connections(index)
test, edges = london_graph(stations, connections)
slay = build_line_map(connections)
#random_node_test(test, stations, connections)





#show the difference in the number of nodes in the shortest path based on lines 
#x axis is path length, y axis is the number of lines/transfers 
def compare_path_length_vs_transfers(graph, station_pos, connections, num_trials=300):
    num_nodes = graph.number_of_nodes()
    line_map = build_line_map(connections) #line map to use to see which lines used
    path_lengths = []
    transfer_cts = []

    while num_trials > 0:
        src = random.randint(0, num_nodes - 1)
        dst = random.randint(0, num_nodes - 1)
        if src == dst: #sjip if same statiuon
            continue

        heuristic = build_heuristic_dict(station_pos, dst)
        pred, path = a_star(graph, src, dst, heuristic)
        #get the length of the path and the number of transfers 
        path_length = len(path)
        num_transfers = len(num_lines_used_in_path(path, line_map))
        path_lengths.append(path_length)
        transfer_cts.append(num_transfers)
        num_trials -= 1

    #use a scatter plot 
    plt.figure(figsize=(8, 5))
    plt.scatter(path_lengths, transfer_cts, color='pink', label='Trials')
    #trned line
    z = np.polyfit(path_lengths, transfer_cts, 1)  #linear line ,d egree 1
    p = np.poly1d(z)
    plt.plot(sorted(path_lengths), p(sorted(path_lengths)), color='red', linestyle='--', label='Trend Line')
    plt.xlabel("Path Length (Number of Stations Go To)")
    plt.ylabel("Number of Line Transfers")
    plt.title("Path Length vs Number of Line Transfers Using A*")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return 


compare_path_length_vs_transfers(test, stations, connections)















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
        elif (v, u) in line_map:
            used_lines.add(line_map[(v, u)])
    return used_lines



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
        _, path = dijkstra2(graph, src, dest)
        #get the number of lines used/ transfers and which linse 
        lines_used = lines_used_in_path(path, line_map)
        #add all data to results as dicts per list item
        results.append({ "source": src, "destination": dest, "path": path, "lines_used": list(lines_used), "num_lines": len(lines_used) })
        trials_done += 1

    return results

# # #build the line map to see
# line_map = build_line_map(id_to_index)
# #print results 
# results = experiment_random_paths_with_lines(graph, station_pos, line_map)
# for iteration in results:
#     print(f"{iteration['source']} → {iteration['destination']}, lines: {iteration['lines_used']}, transfers: {iteration['num_lines']}")




#treat if there is a path from a-b then there is same path from b-a in normal subway
#professor said this is okay
#but i dont actually add it in our graph we build bc it says to make it directed but in terms of lines adjacent we count both ways like this 
