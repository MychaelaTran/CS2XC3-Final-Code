import heapq
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

    return station_pos, station_names


# slay1 = read_stations()
# print(slay1)

def read_connections():
    #read the connections (edges) and we return a list a lsit of (st 1, st2, time)
    
    connected_stations = []
    with open("london_connections.csv", 'r') as file: 
        csvreader = csv.DictReader(file) 
        next(csvreader)
        for row in csvreader:
            st1 = int(row["station1"])
            st2 = int(row["station2"])
            time = int(row["time"])
            connected_stations.append((st1, st2, time))
    return connected_stations

slay2 = read_connections()
print(slay2)


def london_graph(station_pos, connected_stations):
    #build
    return 