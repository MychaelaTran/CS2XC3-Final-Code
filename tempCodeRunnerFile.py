    if (station1, station2) in connected_stations.keys():
                    weight = connected_stations[(station1, station2)]
                    G.add_edge(