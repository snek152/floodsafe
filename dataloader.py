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


def random_lat_lon(n=1, lat_min=-90., lat_max=90., lon_min=-180., lon_max=180.):
    """
    this code produces an array with pairs lat, lon
    """
    lat = np.random.uniform(lat_min, lat_max, n)
    lon = np.random.uniform(lon_min, lon_max, n)

    return np.array(tuple(zip(lat, lon)))


def data_loader():
    lineCount = 0
    with open("processed_weather_data.txt") as file:
        for line in file:
            lineCount += 1
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
            yield weather_data, pathname, iwv
    # INCOMPLETE STUFF
    # with open("processed_weather_data.txt") as file:
    #     i = 0
    #     while i < lineCount:
    #         lat, long = tuple(zip(np.random.uniform(-90., 90., 1),
    #                               np.random.uniform(-180., 180., 1)))[0]
    #         iwv = np.random.uniform(0, 100)
    #         year = np.random.randint(2001, 2021)
    #         month = np.random.randint(1, 13)
    #         day = np.random.randint(1, 29)
    #         hour = np.random.randint(0, 24)
    #         date = datetime.datetime(year, month, day, hour)


def save_data():
    df = pd.DataFrame(
        columns=["image", "lat", "long", "generationtime_ms", "utc_offset_seconds", "timezone", "elevation", "time", "temperature_2m", "iwv", "ar"])
    with open("processed_weather_data.txt") as file:
        count = 0
        for data, image, iwv in data_loader():
            df = df.append({"image": image, "lat": data["latitude"], "long": data["longitude"], "generationtime_ms": data["generationtime_ms"], "utc_offset_seconds": data["utc_offset_seconds"],
                            "timezone": data["timezone"], "elevation": data["elevation"], "time": data["hourly"]["time"], "temperature_2m": data["hourly"]["temperature_2m"], "iwv": iwv, "ar": 1}, ignore_index=True)
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
