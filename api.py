import requests
import json
import dotenv
import os
import csv
import datetime

dotenv.load_dotenv(".env")






# idk how to get the key i think there's a key needed cuz otherwise the data only goes 
# up to the past month but it doesn't offer a key anywhere??

def get_weather_data(year, month, day, lat, lng):
    start_date = datetime.datetime(year, month, day)
    end_date = start_date + datetime.timedelta(days=1)
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lng}&start_date={start_date.strftime('%Y-%m-%d')}&end_date={end_date.strftime('%Y-%m-%d')}&hourly=temperature_2m"
    response = requests.get(url)
    return response.json()



    


# commenting so that i dont mess up!
# def get_weather_data():
#     url = " https://api.ambeedata.com/weather/history/by-lat-lng?lat=12&lng=77&from=2023-01-20 00:00:00&to=2023-01-21 00:00:00"
#     response = requests.get(url, headers={
#         "x-api-key": os.environ.get("RAPID_API_KEY"),
#         "Content-Type": "application/json"
#     })

#     return response.json()


# print(get_weather_data())
# download_image()
# print(get_data())
