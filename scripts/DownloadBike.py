import overpass
import networkx as nx
from tqdm import tqdm
from pyproj import Transformer
import xml.etree.ElementTree as ET
from shapely.geometry import LineString
from scripts.Graph_func import city_to_files

class DownloadBike():
    def __init__(self, where: dict) -> None:
        self.G = None
        self.city_download = self.check_city_name_for_download(where['city'])
        self.city = where['city']
        self.where = where
        self.osm_type = 'bike'

    def check_city_name_for_download(self, city_name):
        if city_name == 'New York City':
            return 'City of New York'
        elif city_name == "Washington, D.C.":
            return 'Washington'
        else:
            return city_name
    
    def download(self):
        network = nx.MultiGraph()
        tree = self.download_osm_bike_data(self.city_download)
        if len(tree.findall('way')) > 0:
            network = self.create_network(tree = tree, network = network)
        print(f"{self.city} has {network.number_of_nodes()} nodes and {network.number_of_edges()} edges")
        self.G = network

    def save_network(self):
        city_to_files(G = self.G, city = self.city, osm_type = self.osm_type, stage = 'raw')

    def download_osm_bike_data(self, city):
        api = overpass.API(timeout = 240)

        # fetch all ways and nodes
        result = api.get(f"""
                        area["name"="{city}"] -> .a;
                        (
                        way["highway" = "cycleway"]["area" != "yes"](area.a);
                        );
                        (._;>>;);
                        out geom;
                        >;
                        """, responseformat="xml")
        tree = ET.ElementTree(ET.fromstring(result))

        return tree

    def get_nodes(self, tree):
        """
        Get all the stations of a relation,
        returns a dict with list, with a 
        stations (point, osm_id)
        """
        node_order = {} #key = rel_id, value = stations nodes
        for rel in tree.findall('way'):
            nodes = []
            way_id = int(rel.attrib['id'])

            #Get nodes of way
            for node in rel.findall('nd'):
                #Get node ids
                lon = float(node.attrib['lon'])
                lat = float(node.attrib['lat'])
                nodes.append([(lon,lat), int(node.attrib['ref'])])
            node_order[way_id] = nodes
        return node_order

    def line_lenght(self, line):
        # Transformer to convert from WGS84 to EPSG:3857 (meters)
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        projected_line = LineString([transformer.transform(*coord) for coord in line.coords])
        length_in_meters = projected_line.length
        return length_in_meters

    def create_network(self, tree, network = nx.MultiGraph):
        node_order = self.get_nodes(tree)
        great_graph = network
        
        #Plot graph
        for way_id in tqdm(node_order.keys(), desc= 'Buiding Lines'):
            nodes = node_order[way_id]
            u = nodes[0]
            v = nodes[-1]

            great_graph.add_node(u[1], x = u[0][0], y = u[0][1])
            great_graph.add_node(v[1], x = v[0][0], y = v[0][1])
            
            coords = [i[0] for i in nodes]

            line = LineString(coords)
            great_graph.add_edge(u_for_edge = u[1], 
                                 v_for_edge = v[1], 
                                 osmid = way_id, 
                                 geometry = line, 
                                 length= self.line_lenght(line))
                        
        return great_graph