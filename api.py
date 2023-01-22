import requests
import json
import dotenv
import os
import csv

dotenv.load_dotenv(".env")


def download_image():
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={400},{-122.0070246}&key={os.environ.get('API_KEY')}&size={600}x{600}&zoom={18}&maptype=satellite"
    response = requests.get(url)
    with open("image.png", "wb") as f:
        f.write(response.content)


def get_weather_data(year, month, day, hour, lat, lng):
    url = f"https://api.ambeedata.com/weather/history/by-lat-lng?lat={lat}&lng={lng}&from={year}-{month}-{day} {hour}:00:00&to={year}-{month}-{day} {hour}:00:00"
    response = requests.get(url, headers={
        "x-api-key": os.environ.get("RAPID_API_KEY"),
        "Content-Type": "application/json"
    })

    return response.json()


weather_data = []
with open("weather_data.txt") as file:
    lines = file.readlines()[:100] #starting slow
    for line in lines:
        numbers = line.split()
        year, month, day, hour, lat, lng = int(numbers[1]), int(numbers[2]), int(numbers[3]), int(numbers[4]), float(numbers[5]), float(numbers[6])
        weather_data.append(get_weather_data(year, month, day, hour, lat, lng))

print("loading successful")


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
