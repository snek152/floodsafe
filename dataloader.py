import api
import scraper
import datetime
import operator
import pandas as pd
import cv2
import os
from typings import APIResponse
import numpy as np
import matplotlib.pyplot as plt


def data_loader():
    with open("processed_weather_data.txt") as file:
        for line in reversed(file.readlines()):
            numbers = line.split()
            year, month, day, hour, lat, lng, iwv = int(numbers[1]), int(numbers[2]), int(
                numbers[3]), int(numbers[4]), float(numbers[5]), float(numbers[6])-360, float(numbers[10])
            if year < 2001:
                break
            date = datetime.datetime(year, month, day, hour)
            date_string = date.strftime("%Y-%m-%d T%H:%M:%SZ")
            pathname = f"images/image_{year}_{month}_{day}_{hour}_{lat}_{lng}.png"
            print(lat, lng, iwv)
            weather_data = api.get_weather_data(year, month, day, lat, lng)
            scraper.save_image(
                lat, lng, zoom=8, date=date_string, pathname=pathname)
            yield weather_data, pathname


def save_data():
    df = pd.DataFrame(
        columns=["image", "lat", "long", "generationtime_ms", "utc_offset_seconds", "timezone", "elevation", "time", "temperature_2m"])
    with open("processed_weather_data.txt") as file:
        count = 0
        for data, image in data_loader():
            df = df.append({"image": image, "lat": data["latitude"], "long": data["longitude"], "generationtime_ms": data["generationtime_ms"], "utc_offset_seconds": data["utc_offset_seconds"],
                           "timezone": data["timezone"], "elevation": data["elevation"], "time": data["hourly"]["time"], "temperature_2m": data["hourly"]["temperature_2m"]}, ignore_index=True)
            count += 1
            if count >= 2:
                break
    df.to_csv("data.csv")


def show_data():
    df = pd.read_csv("data.csv")
    for index, row in df.iterrows():
        image = cv2.imread(row["image"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.show()


# with open("weather_data.txt") as file:
#     with open("processed_weather_data.txt", "w") as file2:
#         lines = []
#         for line in file:
#             numbers = line.split()
#             year, iwv = int(numbers[1]), float(numbers[10])
#             if year > 2002:
#                 lines.append({"year": year, "iwv": iwv, "line": line})
#         lines.sort(key=operator.itemgetter("iwv", "year"))
#         for line in lines:
#             file2.write(line["line"])
# just learned generator functions!! yayy
# for weather_data in data_loader():
#     print(weather_data)
save_data()
show_data()
