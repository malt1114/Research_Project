from scripts.DownloadBike import DownloadBike
from scripts.DownloadPublicStreets import DownloadPublicStreets

cities = [{"city": "Washington, D.C.", "country": "USA"},
          {'city': 'Bergen', 'country': 'Norway'},
          {'city': 'Oslo', 'country': 'Norway'},
          {'city': 'Trondheim', 'country': 'Norway'},
          {"city": "Portland", "state": "Oregon", "country": "USA"},
          ]

def download_city(place):
    #Get bikelane
    print(f'Getting bike {place["city"]}')
    bike_net = DownloadBike(where = place)
    bike_net.download()
    bike_net.save_network()

    print(f'Getting public streets {place["city"]}')
    public_net = DownloadPublicStreets(where = place)
    public_net.download()
    public_net.save_network()


for i in cities:
    download_city(i)