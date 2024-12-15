import re
import time
import requests
import numpy as np
from selenium import webdriver
from bs4 import BeautifulSoup as bs
from IPython.display import clear_output



def reaalstate_norway(city_url, page_number):
    prices = []
    adresses = []
    sqm = []

    driver = webdriver.Firefox()
    driver.maximize_window()

    for i in range(1, page_number+1):

        # clear print output
        clear_output(wait=True)

        url = f'{city_url}&page={i}'
        session = driver.get(url)
        soup = bs(driver.page_source, 'lxml')

        square_name = 'col-span-2 mt-16 flex justify-between sm:mt-4 sm:block space-x-12 font-bold whitespace-nowrap'
        square = soup.find_all('div', class_=square_name)
        adress_name = 'mt-4 sf-line-clamp-2 sm:order-first sm:text-right sm:mt-0 sm:ml-16 sf-realestate-location'
        adress = soup.find_all('div', class_=adress_name)
        price_name = 'col-span-2 sm:flex sm:items-baseline sm:justify-between'
        price = soup.find_all('div', class_=price_name)

        print(len(square), len(adress), len(price))

        for idx, i in enumerate(square):
            sqm.append(i.text.strip().split('m')[0])
            print(idx, i.text.strip().split('m'))

        while len(adress) > len(square):
            adress = adress[1:]
            print(len(adress))

        for idx, i in enumerate(adress):
            adresses.append(i.text)
            print(idx, i.text)

        while len(price) > len(square):
            price = price[1:]
        for idx, i in enumerate(price):
            prices.append(i.text.strip().split('kr')[0])
            print(idx, i.text.strip().split('kr')[0])


        print(len(prices), len(adresses), len(sqm))
        assert len(prices) == len(adresses) == len(sqm)
        time.sleep(3)

    driver.quit()

    return prices, adresses, sqm

def get_lat_long(address, loc):
    try:
        location = loc.geocode(address)
        return location.latitude, location.longitude
    except:
        print(f'error: {address}')
        return np.nan, np.nan


def get_listings_us(api_key, listing_url):
    url = "https://app.scrapeak.com/v1/scrapers/zillow/listing"

    querystring = {
        "api_key": api_key,
        "url":listing_url
    }

    return requests.request("GET", url, params=querystring)


def clean_us(data):
    #Filter based on portland and DC
    data = data[data['address'].str.contains('Portland') | data['address'].str.contains('Washington')]
    #Get city
    data['city'] = data['address'].apply(lambda x: 'Portland' if 'Portland' in x else 'Washington')

    #Rename lot, lan
    data = data.rename({'hdpData.homeInfo.latitude': 'lat',
                                'hdpData.homeInfo.longitude': 'lon'}, axis = 1)
    data['lat'] = data['lat'].apply(str)
    data['lon'] = data['lon'].apply(str)
    data['lat_long'] = "("+data['lat']+','+data['lon']+")"
    
    #get price pr sqm
    data['price'] = data['price'].apply(lambda x: int(re.sub('[^0-9]','', x)))
    data['area'] = data['area']*0.09290304
    data['price_per_sqm'] = data['price']/ data['area'] 
    data['price_per_sqm'] = data['price_per_sqm'].apply(lambda x: round(x, 2))

    #Select columns
    data = data[['city', 'price_per_sqm', 'lat_long']]

    return data


def clean_norway(data):
    data['price'] = data['price'].str.replace('Totalpris:', '').str.replace(chr(160), '').str.strip()
    data['area'] = data['area'].str.replace(chr(160), '').str.strip()
    #Make sure price and area it is numbers
    data['price_is_int'] = data['price'].apply(lambda x: x.isdigit())
    data = data[data['price_is_int']]
    data['price'] = data['price'].apply(int)

    data['area_is_int'] = data['area'].apply(lambda x: x.isdigit())
    data = data[data['area_is_int']]
    data['area'] = data['area'].apply(int)

    data['price_per_sqm'] = data['price']/data['area']
    data['price_per_sqm'] = data['price_per_sqm'].apply(lambda x: round(x, 2))
    data = data[['city', 'price_per_sqm', 'lat_long']]

    return data
    