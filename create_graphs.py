# Import necessary libraries
import networkx as nx
import sys, os, re, numpy as np, shapely, pandas as pd, matplotlib.pyplot as plt, geopandas as gpd
from shapely.geometry import LineString, Polygon, Point, MultiPoint
from shapely import voronoi_polygons, intersection
from joblib import Parallel, delayed
from tqdm import tqdm

# Import custom scripts
sys.path.append('scripts')
from CleanGraph import *
from Graph_func import *


# Define bounding polygons for specific cities
oslo = Polygon([[1182677.03213907,8374160.03311788],
                            [1182677.03213907,8392372.74749964],
                            [1206843.96504923,8392372.74749964],
                            [1206843.96504923,8374160.03311788],
                            [1182677.03213907,8374160.03311788]])

bergen = Polygon([[584007.04835709, 8473059.12383293],
                            [584007.04835709, 8497599.81535933],
                            [599806.30829403, 8497599.81535933],
                            [599806.30829403, 8473059.12383293],
                            [584007.04835709, 8473059.12383293]])

trondheim = Polygon([[1149977.86683372, 9195614.82799802],
                            [1149977.86683372, 9213470.91227339],
                            [1167358.98164311, 9213470.91227339],
                            [1167358.98164311, 9195614.82799802],
                            [1149977.86683372, 9195614.82799802]])

washington = Polygon([[-8615612.67260216,  4677216.28223841],
                [-8615612.67260216,  4742711.43005719],
                [-8549179.43612082,  4742711.43005719],
                [-8549179.43612082,  4677216.28223841],
                [-8615612.67260216,  4677216.28223841]])

portland = Polygon([[-13676784.52437161,5668639.00533991],
                [-13676784.52437161,5735801.6096526 ],
                [-13621546.28832918,5735801.6096526 ],
                [-13621546.28832918,5668639.00533991],
                [-13676784.52437161,5668639.00533991]]
                )

poly_dic = {'Oslo': oslo, 'Bergen': bergen, 'Trondheim': trondheim, 'Washington': washington, 'Portland': portland}

def make_combined_csv(dir_path: str, export_path: str) -> None:
    # Get a list of all csv files in the directory
    csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
    dataframes = []

    for file in tqdm(csv_files):
        # Extract year and month from filename
        # print(file)
        year, month = file.split('.')[0].split('_')

        # Read csv file into a dataframe
        df = pd.read_csv(os.path.join(dir_path, file))

        # Add a new column for the month and year
        df['month'] = month
        df['year'] = year

        # Check if the DataFrame is not empty or does not contain only NaN values (avoid warning when concatenating dfs)
        if not df.empty and not df.isna().all().all():
            dataframes.append(df)

    # Concatenate all dataframes into a single dataframe
    if dataframes:
        combined_df = pd.concat(dataframes)

        # Write the combined dataframe to a new csv file in the export directory
        combined_df.to_csv(f'{export_path}/preprocessed_bike_rides.csv', index=False)
    # functions to create dataframes for stations and rides separately, can be used to create bike network

def create_stations_gdf(df: pd.DataFrame, crs_in: int = 4326, crs_out: int = 3857) -> gpd.GeoDataFrame:
    df = df.copy()  # Create a copy to avoid changing the original DataFrame

    # Create unique dataframes for start and end stations
    start_stations = df[
        [
            "start_station_id",
            "start_station_name",
            "start_station_description",
            "start_station_latitude",
            "start_station_longitude",
            "month",
            "year",
        ]
    ].drop_duplicates()
    end_stations = df[
        [
            "end_station_id",
            "end_station_name",
            "end_station_description",
            "end_station_latitude",
            "end_station_longitude",
            "month",
            "year",
        ]
    ].drop_duplicates()

    # Rename columns for uniformity
    start_stations.columns = [
        "station_id",
        "station_name",
        "station_description",
        "latitude",
        "longitude",
        "month",
        "year",
    ]
    end_stations.columns = [
        "station_id",
        "station_name",
        "station_description",
        "latitude",
        "longitude",
        "month",
        "year",
    ]

    # Concatenate the dfs and drop duplicates
    stations = pd.concat([start_stations, end_stations]).drop_duplicates()
    
    # Create GeoDataFrame
    gdf_stations = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations.longitude, stations.latitude),
        crs=crs_in,
    )
    gdf_stations = gdf_stations.to_crs(epsg=crs_out)

    return gdf_stations

def create_rides_gdf(df: pd.DataFrame, crs_in: int = 4326, crs_out: int = 3857) -> gpd.GeoDataFrame:
    df = df.copy()  # Create a copy to avoid changing the original DataFrame
    
    # Create a new DataFrame with necessary columns
    df_rides = df[
        [
            "start_station_id",
            "end_station_id",
            "started_at",
            "ended_at",
            "duration",
            "start_station_name",
            "start_station_description",
            "start_station_latitude",
            "start_station_longitude",
            "end_station_name",
            "end_station_description",
            "end_station_latitude",
            "end_station_longitude",
            "month",
            "year",
        ]
    ].copy() # copy to avoid warning

    # Create LineString objects
    df_rides.loc[:, "geometry"] = df_rides.apply(
        lambda row: LineString(
            [
                (row["start_station_longitude"], row["start_station_latitude"]),
                (row["end_station_longitude"], row["end_station_latitude"]),
            ]
        ),
        axis=1,
    )

    # Create GeoDataFrame
    gdf_rides = gpd.GeoDataFrame(df_rides, geometry="geometry", crs=crs_in)
    gdf_rides = gdf_rides.to_crs(epsg=crs_out)
    return gdf_rides

def get_box(city):
    if city.lower() == 'oslo':
        poly = Polygon([[1182677.03213907,8374160.03311788],
                        [1182677.03213907,8392372.74749964],
                        [1206843.96504923,8392372.74749964],
                        [1206843.96504923,8374160.03311788],
                        [1182677.03213907,8374160.03311788]])
        return poly, False
    
    elif city.lower() == 'bergen':
        poly = Polygon([[584007.04835709, 8473059.12383293],
                        [584007.04835709, 8497599.81535933],
                        [599806.30829403, 8497599.81535933],
                        [599806.30829403, 8473059.12383293],
                        [584007.04835709, 8473059.12383293]])
        
        return poly, False
    
    elif city.lower() == 'trondheim':
        poly = Polygon([[1149977.86683372, 9195614.82799802],
                        [1149977.86683372, 9213470.91227339],
                        [1167358.98164311, 9213470.91227339],
                        [1167358.98164311, 9195614.82799802],
                        [1149977.86683372, 9195614.82799802]])

        return poly, False
    
    elif 'washington' in city.lower():
        poly = Polygon([[-8615612.67260216,  4677216.28223841],
                        [-8615612.67260216,  4742711.43005719],
                        [-8549179.43612082,  4742711.43005719],
                        [-8549179.43612082,  4677216.28223841],
                        [-8615612.67260216,  4677216.28223841]])
        return poly, False
    
    elif city.lower() == 'portland':
        poly = Polygon([[-13676784.52437161,5668639.00533991],
                        [-13676784.52437161,5735801.6096526 ],
                        [-13621546.28832918,5735801.6096526 ],
                        [-13621546.28832918,5668639.00533991],
                        [-13676784.52437161,5668639.00533991]]
                        )
        return poly, False
    coord = box['elements'][0]['bounds']

    box = Polygon([(coord['maxlon'],coord['minlat']),
                (coord['maxlon'],coord['maxlat']),
                (coord['minlon'], coord['maxlat']),
                (coord['minlon'], coord['minlat']),
                ])
    #box and to be projected (T/F)
    return box, True

def washington_create_stations_gdf(df: pd.DataFrame, crs_in: int = 4326, crs_out: int = 3857) -> gpd.GeoDataFrame:
    df = df.copy()  # Create a copy to avoid changing the original DataFrame

    # Create unique dataframes for start and end stations
    start_stations = df[
        [
            "start_station_id",
            "start_station_name",
            "start_lat",
            "start_lng",
            "month",
            "year",
        ]
    ].drop_duplicates()
    end_stations = df[
        [
            "end_station_id",
            "end_station_name",
            "end_lat",
            "end_lng",
            "month",
            "year",
        ]
    ].drop_duplicates()

    # Rename columns for uniformity
    start_stations.columns = [
        "station_id",
        "station_name",
        "latitude",
        "longitude",
        "month",
        "year",
    ]
    end_stations.columns = [
        "station_id",
        "station_name",
        "latitude",
        "longitude",
        "month",
        "year",
    ]

    # Concatenate the dfs and drop duplicates
    stations = pd.concat([start_stations, end_stations]).drop_duplicates()

    # Create GeoDataFrame
    gdf_stations = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations.longitude, stations.latitude),
        crs=crs_in,
    )
    gdf_stations = gdf_stations.to_crs(epsg=crs_out)


    return gdf_stations.drop_duplicates(subset='station_id')

def portland_create_stations_gdf(df: pd.DataFrame, crs_in: int = 4326, crs_out: int = 3857) -> gpd.GeoDataFrame:
    df = df.copy()  # Create a copy to avoid changing the original DataFrame

    # Create unique dataframes for start and end stations
    start_stations = df[
        [
        "StartHub",
        "StartLatitude",
        "StartLongitude",
        "month",
        "year",
    ]
    ].drop_duplicates()
    end_stations = df[
        [
        "EndHub",
        "EndLatitude",
        "EndLongitude",
        "month",
        "year",
    ]
    ].drop_duplicates()

    # Rename columns for uniformity
    start_stations.columns = [
        "station_name",
        "latitude",
        "longitude",
        "month",
        "year",
    ]
    end_stations.columns = [
        "station_name",
        "latitude",
        "longitude",
        "month",
        "year",
    ]

    # Concatenate the dfs and drop duplicates
    stations = pd.concat([start_stations, end_stations]).drop_duplicates()

    # Create GeoDataFrame
    gdf_stations = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations.longitude, stations.latitude),
        crs=crs_in,
    )
    gdf_stations = gdf_stations.to_crs(epsg=crs_out)

    return gdf_stations.drop_duplicates(subset=['station_name']).dropna(subset='station_name')

# Class to represent and process a city's bike network
class CityLineGraph:
    def __init__(self, city, bike_stations_df=None, amenity_df=None, pop_density=None, geo_data=None):
        # Initialize attributes
        self.city = city
        self.bike_stations_df = bike_stations_df
        self.amenity_df = amenity_df
        self.box = get_box(city)[0]
        self.pop_density = pop_density
        self.G = None
        self.lg = None  # Line graph representation
        self.feats = None  # Features DataFrame
        self.geo_data = geo_data

    # Load the city's graph from data
    def load_g(self, city):
        try:
            G = load_city_graph(city, 'public_streets', stage='deep_clean')  # Attempt deep clean
        except:
            G = load_city_graph(city, 'public_streets', stage='clean')  # Fall back to regular clean
        self.G = nx.Graph(G)

    # Assign attributes like stations and amenities to the graph
    def assign_attributes(self, stations_df, amenity_df):
        # Assign stations to edges
        G = self.assign_edges(self.G, stations_df, self.box)
        
        # Process amenities as GeoDataFrame
        city_amenities = gpd.GeoDataFrame(amenity_df)
        city_amenities['geometry'] = city_amenities.apply(lambda row: list_coord_to_geo(row['geometry'], row['geom_type']), axis=1)
        city_amenities['geometry'] = city_amenities.apply(lambda row: row['geometry'].centroid if row['geom_type'] == 'Polygon' else row['geometry'], axis=1)
        city_amenities = city_amenities.to_crs(epsg=3857)

        # Assign amenities and population density to edges
        for amenity in tqdm(city_amenities.amenity.unique(), total=len(city_amenities.amenity.unique())):
            G = self.assign_other_attributes(G, 
                                            city_amenities.query(f'amenity == "{amenity}"').geometry,
                                            city_amenities.query(f'amenity == "{amenity}"').amenity,
                                            amenity, self.box)
        G = self.assign_other_attributes(G, self.pop_density.geometry, self.pop_density.population, 'population', self.box, True)
        G = self.assign_other_attributes(G, self.geo_data.geometry, self.geo_data['price_per_sqm'], 'price_per_sqm', self.box, True)
        self.G = G

    # Write the graph's line graph features to a file
    def write_graph(self):
        lg_feats = [t[1] for t in list(self.lg.nodes(data=True))]
        feats = pd.DataFrame(lg_feats).fillna(0)

        # Convert columns to appropriate types
        for column in feats.columns:
            try:
                feats[column] = feats[column].astype(float)
            except:
                feats[column] = feats[column].astype(str)

        # Save the features as a Parquet file
        feats.drop(columns=['geometry', 'geom_type']).to_parquet(f'data/{self.city}/linegraph.parquet', index=False)

    def assign_other_attributes(self, G, points, point_ids, feat_name: str, box=None, area=False):
        # Convert the graph into a DataFrame and drop missing 'to' and 'from' columns
        gdf = graph_to_dataframe(G).dropna(subset=['to', 'from'])
        
        # Iterate through points and their IDs
        for i, j in zip(points, point_ids):
            # Print point and ID if ID is None
            if isinstance(j, type(None)):
                print(i, j)
            # Skip if the point is outside the bounding box
            if not shapely.contains(box, i):
                continue
            if area:
                # If attribute is spatial (e.g., population density), assign it to edges within hexagons
                for k in zip(gdf.geometry, gdf['from'], gdf['to']):
                    if shapely.contains(i, k[0]):  # Check if hexagon contains edge
                        if feat_name not in G[k[1]][k[2]]:
                            G[k[1]][k[2]][feat_name] = [j]
                        else:
                            G[k[1]][k[2]][feat_name].append(j)
                    elif shapely.crosses(i, k[0]):  # Check if hexagon crosses edge
                        if feat_name not in G[k[1]][k[2]]:
                            G[k[1]][k[2]][feat_name] = [j]
                        else:
                            G[k[1]][k[2]][feat_name].append(j)
            else:
                # Assign attribute to the nearest edge
                edge, distance, node_pair = self.find_nearest_edge(gdf.geometry, i, gdf['from'], gdf['to'], box)
                try:
                    if feat_name not in G[node_pair[0]][node_pair[1]]:
                        G[node_pair[0]][node_pair[1]][feat_name] = 1
                    else:
                        G[node_pair[0]][node_pair[1]][feat_name] += 1
                except:
                    continue
        return G

    def assign_edges(self, G, point_df, box):
        # Convert the graph into a DataFrame and drop missing 'from' and 'to' columns
        gdf = graph_to_dataframe(G).dropna(subset=['from', 'to'])
        
        # Replace missing values in the points DataFrame with 'None'
        point_df.fillna('None', inplace=True)
        
        # Add a station ID column if it doesn't exist
        if 'station_id' not in point_df.columns:
            point_df['station_id'] = point_df.index
        
        # Iterate through points and assign them to the nearest edge
        for i in tqdm(point_df.iterrows(), total=len(point_df)):
            if not shapely.contains(box, i[1].geometry):
                continue
            # Find the nearest edge
            edge, distance, node_pair = self.find_nearest_edge(gdf.geometry, i[1].geometry, gdf['from'], gdf['to'], box)
            # Assign station information to the edge
            try:
                G[node_pair[0]][node_pair[1]]['station'] = i[1].station_name
                G[node_pair[0]][node_pair[1]]['station_id'] = i[1].station_id
            except:
                print(node_pair, flush=True)
        return G

    def make_list_attributes_mean(self, G, attribute):
        # Calculate the mean for list-type attributes
        for i in tqdm(my_city.G.edges(data=True)):
            if i[1] == {}:
                continue
            if attribute in i[2]:
                if isinstance(i[2][attribute], list):  # Check if the attribute is a list
                    i[2][attribute] = np.mean(i[2][attribute])  # Replace list with its mean value

    def find_nearest_edge(self, linestrings, point, from_node, to_node, box=None):
        # Initialize variables to find the closest edge
        shortest_distance = float('inf')
        closest_edge = None
        node_pair = None
        
        # Check if the point is within the bounding box
        if shapely.contains(box, point):
            for linestring, n1, n2 in zip(linestrings, from_node, to_node):
                # Calculate the distance between the point and the edge
                distance = linestring.distance(point)
                if distance < shortest_distance:
                    shortest_distance = distance
                    closest_edge = linestring
                    node_pair = (n1, n2)
            return closest_edge, shortest_distance, node_pair

    def construct_line_graph(self):
        # Convert the graph into its line graph representation
        H = nx.line_graph(self.G)
        H.add_nodes_from((node, self.G.edges[node]) for node in H)
        H.nodes(data=True)
        self.lg = H  # Store the line graph

    def get_listings(self, city, file_path, bounding_box):
        # Read the listings data
        data = pd.read_csv(file_path)
        # Filter listings for the specified city and valid coordinates
        data = data[data['city'] == city.capitalize()]
        data = data[data['lat_long'] != "(nan, nan)"]
        data = data[data['lat_long'] != "(nan,nan)"]
        
        # Convert lat_long strings into float values
        data['lat_long'] = data['lat_long'].apply(lambda x: x.replace('(', '').replace(')', '').split(','))
        data['geometry'] = data['lat_long'].apply(lambda x: Point(float(x[1]), float(x[0])))
        
        # Convert to GeoDataFrame and reproject
        data_geo = gpd.GeoDataFrame(data, geometry='geometry', crs='EPSG:4326')
        data_geo = data_geo.to_crs(crs='EPSG:3857')
        self.get_poly_and_cut(data_geo, bounding_box)

    def get_poly_and_cut(self, data, bounding_box):
        # Filter data points within the bounding box
        data['keep'] = data['geometry'].apply(lambda x: bounding_box.contains(x))
        data = data[data['keep']]
        
        # Create Voronoi polygons from the data points
        points = MultiPoint(data['geometry'].to_list())
        polygons = voronoi_polygons(points)
        
        # Assign listing information to Voronoi polygons
        poly_data = []
        for i in list(polygons.geoms):
            temp = data.copy()
            temp['contains'] = temp['geometry'].apply(lambda x: i.contains(x))
            temp = temp[temp['contains']]
            poly_data.append([temp.iloc[0]['price_per_sqm'], i])
        
        # Create a GeoDataFrame with polygon geometries
        poly_geo = gpd.GeoDataFrame(poly_data, columns=['price_per_sqm', 'geometry'], crs='EPSG:3857')
        poly_geo['geometry'] = poly_geo['geometry'].apply(lambda x: intersection(bounding_box, x))
        self.geo_data = poly_geo

    def get_pop_density(self, city):
        # Load population density data
        pop_density = gpd.read_file(f'data/{city}/population_density.gpkg')
        # Normalize population values and reproject
        pop_density['population'] = pop_density['population'].apply(lambda x: x / 0.61)
        pop_density = pop_density.to_crs(epsg=3857)
        self.pop_density = pop_density  # Store the population density GeoDataFrame

    def read_graph(self):
        # Load the graph or create a new one
        self.load_g(self.city)
        self_loops = list(nx.selfloop_edges(self.G))  # Remove self-loops
        self.G.remove_edges_from(self_loops)
        Gcc = sorted(nx.connected_components(self.G), key=len, reverse=True)
        self.G = self.G.subgraph(Gcc[0])  # Keep the largest connected component
        print('Self Loops Removed', flush=True)
        self.lg = nx.line_graph(self.G)
        self.feats = pd.read_parquet(f'data/{self.city}/linegraph.parquet')
        node_dict = {tuple(self.feats.iloc[idx, :2]): self.feats.iloc[idx, 2:].to_dict() for idx in range(len(self.feats))}
        nx.set_node_attributes(self.lg, node_dict)

    def main(self):
        # Main pipeline for processing
        self.load_g(self.city)
        self.get_pop_density(self.city)
        self.get_listings(self.city, 'data/listings/clean/listings.csv', self.box)
        self.assign_attributes(self.bike_stations_df, self.amenity_df)
        self.make_list_attributes_mean(self.G, 'price_per_sqm')
        self.make_list_attributes_mean(self.G, 'population')
        self.construct_line_graph()
        self.write_graph()


city = str(sys.argv[1]).lower()
# city = 'trondheim'

print(city, flush = True)

dir_path = f'data/bike_sharing_data/{city}'
export_path = f'data/bike_sharing_data/{city}/processed'

try:
    combined_df = pd.read_csv(f'{export_path}/preprocessed_bike_rides.csv')
except:
    make_combined_csv(dir_path, export_path)


# Load the combined DataFrame with functions
combined_df = pd.read_csv(f'{export_path}/preprocessed_bike_rides.csv')

if city == 'portland':
    stations_df = portland_create_stations_gdf(combined_df)
elif city == 'washington':
    stations_df = washington_create_stations_gdf(combined_df)
else:
    rides_df = create_rides_gdf(combined_df)
    stations_df = create_stations_gdf(combined_df)

city_amenities = gpd.GeoDataFrame(pd.read_json(f'data/{city}/{city}_amenities.json'))


my_city = CityLineGraph(city, stations_df, city_amenities)
my_city.main()
