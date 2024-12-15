from scripts.DeepClean import DeepCleanGraph
from scripts.CleanGraph import CleanGraph
from joblib import Parallel, delayed
import time 

N_JOBS = 3

cities = [{"city": "Washington", "country": "USA"},
          {"city": "Portland", "state": "Oregon", "country": "USA"},
          {'city': 'Oslo', 'country': 'Norway'},
          {'city': 'Bergen', 'country': 'Norway'},
          {'city': 'Trondheim', 'country': 'Norway'},
          ]


#Clean bikes
print('CLEANING BIKE NETWORK', flush = True)
clean_bike = [CleanGraph(where = place, osm_type = 'bike') for place in cities]
with Parallel(n_jobs=N_JOBS, verbose=-1) as parallel:
        #Prepare the jobs
        delayed_funcs = [delayed(lambda x:x.clean_graph())(city) for city in clean_bike]
        #Runs the jobs in parallel
        parallel(delayed_funcs)

time.sleep(30)

#Deep clean bikes
print('DEEP CLEANING BIKE NETWORK', flush = True)
deep_clean_bike = [DeepCleanGraph(where = place, osm_type = 'bike') for place in cities]
with Parallel(n_jobs=N_JOBS, verbose=-1) as parallel:
        #Prepare the jobs
        delayed_funcs = [delayed(lambda x:x.clean_graph())(city) for city in deep_clean_bike]
        #Runs the jobs in parallel
        parallel(delayed_funcs)

time.sleep(30)

#Clean plublic streets
print('CLEANING STREETS NETWORK', flush = True)
clean_streets = [CleanGraph(where = place, osm_type = 'public_streets') for place in cities]
with Parallel(n_jobs=N_JOBS, verbose=-1) as parallel:
        #Prepare the jobs
        delayed_funcs = [delayed(lambda x:x.clean_graph())(city) for city in clean_streets]
        #Runs the jobs in parallel
        parallel(delayed_funcs)

time.sleep(30)

#Deep clean streets
print('DEEP CLEANING STREETS NETWORK', flush = True)
deep_clean_streets = [DeepCleanGraph(where = place, osm_type = 'public_streets') for place in cities]
with Parallel(n_jobs=N_JOBS, verbose=-1) as parallel:
        #Prepare the jobs
        delayed_funcs = [delayed(lambda x:x.clean_graph())(city) for city in deep_clean_streets]
        #Runs the jobs in parallel
        parallel(delayed_funcs)