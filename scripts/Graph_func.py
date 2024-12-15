import os
import re
import json
import pyproj
import pandas as pd
import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely import get_coordinates
from shapely.ops import transform
from shapely.geometry import Point, LineString, Polygon, MultiPolygon


def load_city_graph(city: str, osm_type:str, stage:str):
    city = city.lower().replace(',', '').replace('.','').replace(' ', '_')
    if "washington" in city:
        city = "washington"
    
    G = nx.MultiGraph()

    base_path = f"data/{city}/{stage}/"
    
    #Add the nodes
    file = open(base_path + f"{city}_{osm_type}_node_att.txt", "r")
    while True:
        content=file.readline()
        if not content:
            break
        d = content.split(' ', 1)
        attr = re.sub("((?<!{)(?<!,\s)(?<!: ))\"((?!:)(?!,\s)(?!}))", "", d[1]).replace(': None', ': ""').replace("\\", '')
        attr = json.loads(attr)
        if 'geometry' in attr.keys():
            attr['geometry'] = list_coord_to_geo(attr['geometry'], attr['geom_type'])
        G.add_node(int(d[0]), **attr)
    file.close()
        
    #Add the edges
    file = open(base_path + f'{city}_{osm_type}_edge.txt', "r")
    while True:
        content=file.readline()
        if not content:
            break
        d = content.split(' ', 2)
        attr = json.loads(d[2])
        if 'geometry' in attr.keys():
            attr['geometry'] = list_coord_to_geo(attr['geometry'], attr['geom_type'])
        G.add_edge(int(d[0]), int(d[1]), **attr)
    file.close()

    return G

def load_city_amenities(city, clean = False):
    """
    Loads either the raw data or the compresed data
    """
    if 'washington' in city.lower():
        city = 'Washington'
    city = city.lower()
    if not clean:
        return gpd.read_file(f"data/{city}/{city}_amenities.geojson")
    else:
        data = pd.read_json(f'data/{city}/{city}_amenities.json')
        data = gpd.GeoDataFrame(data)

        data['geometry'] = data.apply(lambda row: list_coord_to_geo(row['geometry'], row['geom_type']), axis = 1)
        data = data.set_geometry('geometry')
        data = data.set_crs('EPSG:3857')
        return data

def make_geom_to_string(geom):
    geom_type = geom.geom_type

    if geom_type != 'MultiPolygon':
        geom = get_coordinates(geom).tolist()
    else:
        multi = []
        for i in range(len(geom.geoms)):
            multi.append(get_coordinates(geom.geoms[i]).tolist())
        geom = multi
    
    return geom, geom_type

def list_coord_to_geo(coord, geom_type):
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
        print(geom_type, 'is unknown')

def download_city_amenities(where: dict):
    """
    Downloads amenities of a city and saves it as a geojson
    _________
    where:dict -> where: dict -> dict containing the placem fx. {"city": "Portland", "state": "Oregon", "country": "USA"}
    _________
    return geopandas dataframe
    """
    city = where['city'].lower().replace(',', '').replace('.','').replace(' ', '_')
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(f"data/{city}"):
        os.chdir(f"data")
        os.makedirs(city)
        os.chdir('..')   
    amenities = ox.features.features_from_place(where, tags = {'amenity':True})
    amenities.to_file(f'data/{city}/{city}_amenities.geojson', driver="GeoJSON")
    return amenities

###################SAVE FUNCTIONS#####################
def city_node_att_to_txt(G: nx.graph, city: str, osm_type:str, stage:str):
    lines = []

    for n in G.nodes(data = True):
        attr = n[1]
        if 'geometry' in attr.keys():
            attr['geometry'], attr['geom_type'] = make_geom_to_string(attr['geometry'])
        lines.append(f"{n[0]} {attr}\n".replace("'", '"'))
    
    path = f'data/{city}/{stage}/{city}_{osm_type}_node_att.txt'

    with open(path, "w", encoding= 'utf8') as file:
        file.writelines(lines)
        file.close()

def city_edgelist_to_txt(G: nx.graph, city: str, osm_type:str, stage:str):
    lines = []
    for e in G.edges(data = True):
        att = {}
        if len(e) > 2:
            att = e[2]
        if 'geometry' in att.keys():
            #Make geometry to a list
            att['geometry'], att['geom_type'] = make_geom_to_string(att['geometry'])

        lines.append(f"{e[0]} {e[1]} {json.dumps(att, ensure_ascii=False)}\n")
    path = f'data/{city}/{stage}/{city}_{osm_type}_edge.txt'
    with open(path, "w", encoding= 'utf8') as file:
        file.writelines(lines)
        file.close()

def city_to_files(G: nx.graph, city: str, osm_type:str, stage:str):
    
    city = city.lower().replace(',', '').replace('.','').replace(' ', '_')
    if "washington" in city:
        city = "washington"
    
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(f"data/{city}"):
        os.chdir(f"data")
        os.makedirs(city)
        os.chdir('..')
    if not os.path.exists(f"data/{city}/raw"):
        os.chdir(f"data/{city}")
        os.makedirs("raw")
        os.chdir('..')
        os.chdir('..')
    if not os.path.exists(f"data/{city}/clean"):
        os.chdir(f"data/{city}")
        os.makedirs("clean")
        os.chdir('..')
        os.chdir('..')
    if not os.path.exists(f"data/{city}/deep_clean"):
        os.chdir(f"data/{city}")
        os.makedirs("deep_clean")
        os.chdir('..')
        os.chdir('..')

    city_edgelist_to_txt(G, city, osm_type, stage)
    city_node_att_to_txt(G, city, osm_type, stage)




################ OTHER FUNCTIONS #####################

def graph_to_dataframe(G:nx.Graph, raw = False):
    data = []

    if raw: 
        for node in G.nodes(data = True):  
            node_data = {}
            if node[1] != {}:
                node_data['from'] = node[0]
                node_data['geometry'] = Point(node[1]['x'], node[1]['y'])
                data.append(node_data)
    else:
        for node in G.nodes(data = True):  
            node_data = {}
            if node[1] != {}:
                node_data['from'] = node[0]
                node_data['geometry'] = node[1]['geometry']
                data.append(node_data)
    #Get edges
    for edge in G.edges(data = True):
        edge_data = edge[2]
        edge_data['from'] = edge[0]
        edge_data['to'] = edge[1]
        data.append(edge_data)

    data = gpd.GeoDataFrame(data, geometry= 'geometry')
    if raw: #Project if raw
        frm = pyproj.CRS('EPSG:4326')
        to = pyproj.CRS('EPSG:3857')
        project = pyproj.Transformer.from_crs(frm, to, always_xy=True).transform
        data['geometry'] = data['geometry'].apply(lambda x: transform(project, x))
    
    data = data.set_crs('EPSG:3857')
    return data