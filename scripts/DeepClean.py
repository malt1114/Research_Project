import json
import os
import re
import pandas as pd
import networkx as nx
import geopandas as gpd
from shapely import get_coordinates, intersection, overlaps
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, MultiLineString, MultiPoint
from scripts.Graph_func import graph_to_dataframe
from shapely.ops import split, unary_union
from scripts.Graph_func import city_to_files, load_city_graph
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


class DeepCleanGraph():
    def __init__(self, where: dict, osm_type:str) -> None:
        self.G = None
        self.city = where['city']
        self.osm_type = osm_type
        self.geometries = None
        self.start_time = time.time()
    
    def clean_graph(self):
        self.G = load_city_graph(city = self.city, osm_type = self.osm_type, stage = 'clean')
        self.print_get_stats(title= f"Cleaning stats for {self.city}", time = 'Before')
        
        #Remove self loops
        self.remove_self_loops()
 
        #Make first clean of non-nesseacary nodes 
        #(reduces the computations of adding intersections)
        new_edges, nodes_to_remove = self.clean_paths()
        self.G = self.add_new_edges(new_edges, self.G)
        self.G = self.remove_nodes(nodes_to_remove, self.G)

        self.print_get_stats(title= f"Cleaning stats for {self.city}", time = 'Before Intersections')
        #Add intersections
        self.add_intersections_to_network()
        self.print_get_stats(title= f"Cleaning stats for {self.city}", time = 'After Intersections')
        
        #Connect if nodes are within distance of 10 meters
        one_degrees = [n for n in self.G.nodes if self.G.degree(n) == 1]
        node_points = graph_to_dataframe(self.G.copy())
        node_points = node_points[node_points['to'].isna()][['from', 'geometry']]
        new_edges = []
        for u in one_degrees:
            new_edges += self.get_new_edges_dist_based(G = self.G, node_points = node_points, u = u, meters = 10)
        self.G = self.add_new_edges(new_edges, self.G)

        #Remove self loops
        self.remove_self_loops()

        #Delete nodes with a degree of 0 or 1 (not bike network)
        if self.osm_type != 'bike':
            self.delete_nodes_one_degree()

        #Make second clean of non-nesseacary nodes 
        #(reduces the computations of adding intersections)
        new_edges, nodes_to_remove = self.clean_paths(multi = True)
        self.G = self.add_new_edges(new_edges, self.G)
        self.G = self.remove_nodes(nodes_to_remove, self.G)
        
        #Remove self loops
        self.remove_self_loops()
        
        #Print final stats and save
        self.print_get_stats(title= f"Cleaning stats for {self.city}", time = 'After')
        city_to_files(G = self.G, city = self.city, osm_type = self.osm_type, stage = 'deep_clean')

    def print_get_stats(self, title, time):
        G = self.G
        components = nx.connected_components(G)
        components = [i for i in list(components)]
        component_sizes = [len(i) for i in components]
        isolated_nodes = nx.isolates(G)

        stats = f"""{time} cleaning {self.city} {self.osm_type}\nNumber of nodes: {len(G.nodes)}\nNumber of edges: {len(G.edges)}\nNumber of connected components: {len(component_sizes)}\nMax size of components: {max(component_sizes)}\nMin size of components: {min(component_sizes)}\nIsolated nodes: {len(list(isolated_nodes))}\n"""
        
        city = self.city.lower()

        f = open(f"data/{city}/{title}.txt", "a")
        f.write(stats)
        f.close()
    
    def add_new_nodes(self, node_list, G):
        new_G = G.copy()
        for i in node_list:
            new_G.add_node(int(i[0]), **{'geometry': i[1]})
        return new_G
    
    def remove_nodes(self, nodes_to_remove, G):
        new_G = G.copy()
        for i in nodes_to_remove:
            new_G.remove_node(i)
        return new_G

    def remove_edges(self, edge_list, G):
        new_G = G.copy()
        edge_list = list(set(edge_list))
        for i in edge_list:
            u = int(i[0])
            v = int(i[1])
            if new_G.has_edge (u, v):
                new_G.remove_edge(u, v)
            else:
                new_G.remove_edge(v, u)
        return new_G

    def add_new_edges(self, edge_list, G):
        new_G = G.copy()
        for i in edge_list:
            new_G.add_edge(int(i[0]), int(i[1]), **{'geometry':i[2]})
        return new_G

    def split_edge_at_points(self, line, points):
        #Split the line into edges
        segments = []
        current_line = line
        #For each point
        for point in points:
            #split line at point
            split_lines = split(current_line, point)
            #if the line were splitted
            if len(split_lines.geoms) >= 2:
                #Add line to segments
                segments.append(split_lines.geoms[0])
                current_line = split_lines.geoms[1]
        #Add last line to segments
        segments.append(current_line)
        return segments

    def add_intersections_to_network(self):
        new_G = nx.MultiGraph()

        # Create a list of edges
        edges = list(self.G.edges(data=True))

        t_0 = time.time()
        # Check for intersections
        for i, (u1, v1, attr1) in enumerate(edges):
            split_by = []
            line1 = attr1['geometry']

            #Get intersecting lines
            for j, (u2, v2, attr2) in enumerate(edges):
                if i != j:
                    line2 = attr2['geometry']
                    if line1.intersects(line2) and line1 != line2:
                        inter = line1.intersection(line2)
                        if isinstance(inter, Point):
                            split_by.append(inter)
                        elif isinstance(inter, MultiPoint):
                            for g in inter.geoms:
                                split_by.append(g)
            
            new_edges = split(line1, MultiPoint(split_by))
            for e in new_edges.geoms:
                #Get start node
                start = e.coords[0]
                start_name = f"{int(start[0])}{int(start[1])}"
                #Get end node
                end = e.coords[-1]
                end_name = f"{int(end[0])}{int(end[1])}"
                
                #add nodes if not in G
                if not new_G.has_node(start_name):
                    new_G.add_node(start_name, geometry = Point(start))
                
                if not new_G.has_node(end_name):
                    new_G.add_node(end_name, geometry = Point(end))
                
                #Add edge
                new_G.add_edge(start_name, end_name, geometry = e)
                    
            if (i % 1000) == 0:
                t_1 = time.time()
                print(f"{i} - {round((t_1-t_0)/1000,2)} - nodes:{new_G.number_of_nodes()}, edges: {new_G.number_of_edges()}", flush = True)
                t_0 = time.time()
                        
        self.G = new_G

    def traverse_until_degree_more_than_two(self, G, node, p, previous):
        neigh = list(G.neighbors(node))
        neigh.remove(previous)
        if len(neigh) == 0:
            p.append(node)
        elif G.degree(neigh[0]) == 2:
            p.append(node)
            self.traverse_until_degree_more_than_two(G, neigh[0], p, node)
        else:
            p.append(node)
            p.append(neigh[0])
            
    def get_two_degree_paths(self, G, node): 
        paths = []
        for n in list(G.neighbors(node)):
            if G.degree(n) == 2:
                p = [node]
                self.traverse_until_degree_more_than_two(G = G, node = n, p = p, previous= node)
                paths.append(p)
        return paths

    def get_new_edges_dist_based(self, G: nx.Graph, node_points:pd.DataFrame, u:int, meters:int):
        new_edges = [] #[u, v, geometry]
        u_geo = node_points[node_points['from'] == u].geometry.item()
        temp = node_points[~node_points['from'].isin(list(G.neighbors(u))+[u])].copy()
        temp['dist'] = temp.geometry.apply(lambda x: u_geo.distance(x))
        temp = temp[temp['dist'] <= meters]
        if temp.shape[0] > 0: 
            for idx, row in temp.iterrows():
                new_edges.append([u, row['from'], LineString([u_geo, row['geometry']])])
        return new_edges

    def delete_nodes_one_degree(self):
        new_G = self.G.copy()
        while True:
            degree_one = [n for n in new_G.nodes if new_G.degree(n) <= 1]
            if degree_one != []:
                for n in degree_one:
                    new_G.remove_node(n)
            else:
                break
        self.G = new_G

    def remove_self_loops(self):
        new_G = self.G.copy()
        new_G.remove_edges_from(nx.selfloop_edges(new_G))
        self.G = new_G

    def clean_paths(self, multi = False):
        G = self.G
        total_paths = []
        for i in [n for n in G.nodes if G.degree(n) > 2]:
            total_paths += self.get_two_degree_paths(G, i)
        
        total_paths = [i for i in total_paths if len(i) > 3]

        new_edges = [] #[u, v, geometry]
        nodes_to_remove = []

        for p in total_paths:
            #Add intermediate nodes, which should be removed
            nodes_to_remove += p[1:-1]
            #Get start point of the first node
            start_point = G.nodes(data=True)[p[0]]['geometry']
            #Set start point to previous point
            previous_point = list(start_point.coords)[0]
            #Add to new geometry
            new_geo = [previous_point]

            for idx in range(len(p)-1):
                #Get edge geo and coordinates
                edge_geo = G[p[idx]][p[idx+1]][0]['geometry']
                edge_geo_coords = list(edge_geo.coords)
                #Get start and begining
                first_point = edge_geo_coords[0]
                last_point = edge_geo_coords[-1]
                
                #If alines with previous point
                if first_point == previous_point:
                    previous_point = edge_geo_coords[-1]
                #If not alines, reverse linestring
                elif last_point == previous_point:
                    previous_point = edge_geo_coords[0]
                    edge_geo_coords.reverse()
                new_geo += edge_geo_coords
            
            new_edges.append([p[0], p[-1], LineString(new_geo)])
            
        return new_edges, list(set(nodes_to_remove))
