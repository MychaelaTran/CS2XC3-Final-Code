import heapq
from part3 import Graph
from part4 import a_star
import csv
import math


class Graph:
    def __init__(self, num_nodes):
        self.graph = {i: [] for i in range(num_nodes)}  # adjacency list
        self.weight = {}

    def add_edge(self, u, v, w):
        self.graph[u].append(v)
        self.weight[(u, v)] = w

    def connected_nodes(self, u):
        return self.graph[u]

    def number_of_nodes(self):
        return len(self.graph)

    def get_graph(self):
        return self.graph

    def get_weights(self):
        return self.weight


#each statuoin bas node in graph
#edge exist if two stations connect4ed
#weigts of edgfes, use lat and long for each station to find the distance traveleed between the two stations
#that distance can servce as teh weight fir a fiven edge
#heruristic use the physical direct distance, not driving distance betweem the srouce adn a given station 
#parameter is string name of csv station file name
def read_stations():
    #we return the station position as a dict[int (latitiude, longititude)] and station name dict[int, (name: stR)]

    station_pos = {}
    station_names = {}
    with open("london_stations.csv", 'r') as file: 
        csvreader = csv.DictReader(file) #read as dict and can access as cokumne name
        next(csvreader) #dont need to read the first inital row 
        for row in csvreader:
            station_id = int(row["id"])
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            name = row["name"]  #using name not display name

            station_pos[station_id] = (lat, lon)
            station_names[station_id] = name

    return station_pos


#slay1 = read_stations()
#print(slay1)

def read_connections():
    #read the connections (edges) and we return a list a lsit of (st 1, st2, time)
    
    connected_stations = {}
    with open("london_connections.csv", 'r') as file: 
        csvreader = csv.DictReader(file) 
        next(csvreader)
        for row in csvreader:
            st1 = int(row["station1"])
            st2 = int(row["station2"])
            time = int(row["time"])
            connected_stations[(st1,st2)] = time
    return connected_stations




def euclidean_dist(station_a, station_b, station_pos):
    (lat1, lon1) = station_pos[station_a]
    (lat2, lon2) = station_pos[station_b]
    return math.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

def london_graph(station_pos, connected_stations):
    num_stations = len(station_pos)
    G = Graph(num_stations)
    
    for (st1, st2) in connected_stations:
        if st1 in station_pos and st2 in station_pos:
            #find eucidean distacne to get the edge weighth 
            dist = euclidean_dist(st1, st2, station_pos)
            G.add_edge(st1, st2, dist)

    return G

connections = read_connections()
stations = read_stations()

test = london_graph(stations, connections)
print("this is test", test.graph)
print("this is wegith", test.weight)