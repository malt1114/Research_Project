import os
import re
import json
import pyproj
import overpass
import networkx as nx
from pyproj import Transformer
from shapely.ops import transform
from shapely import get_coordinates
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from scripts.Graph_func import city_to_files, load_city_graph

class CleanGraph():
    def __init__(self, where: dict, osm_type:str) -> None:
        self.G = None
        self.city = where['city']
        self.osm_type = osm_type
    
    def clean_graph(self):
        self.G = load_city_graph(city = self.city, osm_type = self.osm_type, stage = 'raw')
        print(f"Before cleaning: Size of {self.city} {self.osm_type} has {len(self.G.nodes)} nodes and {len(self.G.edges)} edges", flush= True)
        self.network_pruning(self.city, self.G)
        print(f"After cleaning: Size of {self.city} {self.osm_type} has {len(self.G.nodes)} nodes and {len(self.G.edges)} edges", flush= True)
        
        city_to_files(G = self.G, city = self.city, osm_type = self.osm_type, stage = 'clean')


    def get_box(self):
        """
        Downloads a bounding box for the city or 
        returns the one given by the bike data
        """
        api = overpass.API()

        if self.city.lower() == 'oslo':
            poly = Polygon([[1182677.03213907,8374160.03311788],
                            [1182677.03213907,8392372.74749964],
                            [1206843.96504923,8392372.74749964],
                            [1206843.96504923,8374160.03311788],
                            [1182677.03213907,8374160.03311788]])
            return poly, False
        
        elif self.city.lower() == 'bergen':
            poly = Polygon([[584007.04835709, 8473059.12383293],
                            [584007.04835709, 8497599.81535933],
                            [599806.30829403, 8497599.81535933],
                            [599806.30829403, 8473059.12383293],
                            [584007.04835709, 8473059.12383293]])
            
            return poly, False
        
        elif self.city.lower() == 'trondheim':
            poly = Polygon([[1149977.86683372, 9195614.82799802],
                            [1149977.86683372, 9213470.91227339],
                            [1167358.98164311, 9213470.91227339],
                            [1167358.98164311, 9195614.82799802],
                            [1149977.86683372, 9195614.82799802]])

            return poly, False
        
        elif 'washington' in self.city.lower():
            poly = Polygon([[-8615612.67260216,  4677216.28223841],
                            [-8615612.67260216,  4742711.43005719],
                            [-8549179.43612082,  4742711.43005719],
                            [-8549179.43612082,  4677216.28223841],
                            [-8615612.67260216,  4677216.28223841]])
            return poly, False
        
        elif self.city.lower() == 'portland':
            poly = Polygon([[-13676784.52437161,5668639.00533991],
                            [-13676784.52437161,5735801.6096526 ],
                            [-13621546.28832918,5735801.6096526 ],
                            [-13621546.28832918,5668639.00533991],
                            [-13676784.52437161,5668639.00533991]]
                            )
            return poly, False
        
        elif self.city.lower() == 'helsinki':
            box = api.get("""rel[admin_level=8][name="Helsinki"]; out geom;""", responseformat="json")
        
        elif 'new york' in self.city.lower():
            box = api.get("""rel[admin_level=5][name="City of New York"]; out geom;""", responseformat="json")
        
        elif self.city.lower() == 'vancouver':
            box = api.get("""rel[admin_level=8][name="Vancouver"]; out geom;""", responseformat="json") 

        coord = box['elements'][0]['bounds']

        box = Polygon([(coord['maxlon'],coord['minlat']),
                    (coord['maxlon'],coord['maxlat']),
                    (coord['minlon'], coord['maxlat']),
                    (coord['minlon'], coord['minlat']),
                    ])
        #box and to be projected (T/F)
        return box, True

    def project_coords(self, element):
        """
        Reprojects the geometry
        """
        frm = pyproj.CRS('EPSG:4326')
        to = pyproj.CRS('EPSG:3857')

        project = pyproj.Transformer.from_crs(frm, to, always_xy=True).transform
        return transform(project, element)

    def clean_edge_attr(self, attr):
        """
        This function cleans the edge attributes
        """
        new_attr = {}

        #Get way attributes
        if 'attr_dict' in attr.keys():
            way_attrs = attr['attr_dict']
            way_ids = attr['osmid']
            way_ids = [str(i) for i in way_ids]
            
            #Get all attributes for the individual ways
            all_keys = []
            for i in way_ids:
                all_keys += list(way_attrs[i].keys())
            all_keys = set(all_keys)

            #Merge keys
            for key in all_keys:
                values = set()
                for way in way_ids:
                    if key in list(way_attrs[way].keys()):
                        values.add(way_attrs[way][key])
                values = list(values)
                
                if len(values) == 1:
                    new_attr[key] = values[0]
                else:
                    new_attr[key] = values
        else:
            for key, values in attr.items():
                new_attr['osmid'] = attr['osmid']
                new_attr['length'] = attr['length']
        
        #Project geometries
        if 'geometry' in attr.keys():
            line = attr['geometry']
            new_attr['geometry'] = self.project_coords(line) 
        
        return new_attr

    def network_pruning(self, city: str, G: nx.graph):
        """
        This function cleans the network nodes, 
        edges and their attributes
        """
        box, to_project = self.get_box()
        
        if to_project:
            box = self.project_coords(box)

        pruned_G = nx.MultiGraph()

        nodes_to_keep = []
        nodes_data = []
        for node in G.nodes(data= True):
            node_id = node[0]
            node_attr = node[1]
            #Create point
            node_point = Point(node_attr['x'],node_attr['y'])
            node_point = self.project_coords(node_point)
            node_attr['geometry'] = node_point
            #Check if within box
            if box.contains(node_point):
                nodes_data.append((node_id, node_attr))
                nodes_to_keep.append(node_id)
                pruned_G.add_node(node_id, **node_attr)
        
        #Remove edges
        for edges in G.edges(data = True):
            frm = edges[0]
            to = edges[1]
            if frm in nodes_to_keep and to in nodes_to_keep:
                attr = edges[2]
                new_attr = self.clean_edge_attr(attr)
                pruned_G.add_edge(frm, to, **new_attr)

        self.G = pruned_G

    def make_geom_to_string(self, geom):
        geom_type = geom.geom_type
        #geom = self.project_coords(geom)

        if geom_type != 'MultiPolygon':
            geom = get_coordinates(geom).tolist()
        else:
            multi = []
            for i in range(len(geom.geoms)):
                multi.append(get_coordinates(geom.geoms[i]).tolist())
            geom = multi
        
        return geom, geom_type

    def list_coord_to_geo(self, coord, geom_type):
        if geom_type == 'Point':
            return Point(coord)
        elif geom_type == 'LineString':
            return LineString(coord)
        elif geom_type == 'Polygon':
            return Polygon(coord)
        elif geom_type == 'MultiPolygon':
            polys = [Polygon(i) for i in coord]
            return MultiPolygon(polys)
        else:
            print(geom_type, 'is unknown', flush = True)