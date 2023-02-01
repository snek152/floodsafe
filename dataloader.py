import api
import scraper
import datetime
import operator
import pandas as pd
import cv2
import os
# from typings import APIResponse
import numpy as np
import matplotlib.pyplot as plt


def data_loader_with_ar():
    with open("processed_weather_data.txt") as file:
        for line in file:
            numbers = line.split()
            year, month, day, hour, lat, lng, iwv = int(numbers[1]), int(numbers[2]), int(
                numbers[3]), int(numbers[4]), float(numbers[5]), float(numbers[6])-360, float(numbers[10])
            date = datetime.datetime(year, month, day, hour)
            date_string = date.strftime("%Y-%m-%d T%H:%M:%SZ")
            pathname = f"images/image_{year}_{month}_{day}_{hour}_{lat}_{lng}.png"
            print(lat, lng, iwv)
            if os.path.exists(pathname):
                continue
            weather_data = api.get_weather_data(year, month, day, lat, lng)
            scraper.save_image(
                lat, lng, zoom=8, date=date_string, pathname=pathname)
            yield weather_data, pathname


def data_loader_with_no_ar():
    with open("processed_weather_data.txt") as file:
        for _ in file:
            lat, long = tuple(zip(np.random.uniform(-90., 90., 1),
                                  np.random.uniform(-180., 180., 1)))[0]
            year = np.random.randint(2002, 2023)
            month = np.random.randint(1, 13)
            day = np.random.randint(1, 29)
            hour = np.random.randint(0, 24)
            date = datetime.datetime(year, month, day, hour)

            date_string = date.strftime("%Y-%m-%d T%H:%M:%SZ")
            pathname = f"images/image_{year}_{month}_{day}_{hour}_{lat}_{long}.png"
            weather_data = api.get_weather_data(year, month, day, lat, long)
            scraper.save_image(lat, long, zoom=8,
                               date=date_string, pathname=pathname)
            yield weather_data, pathname


def save_data():
    df = pd.DataFrame(
        columns=["image", "lat", "long", "generationtime_ms", "utc_offset_seconds", "timezone", "elevation", "time", "temperature_2m", "ar"])
    for data, image in data_loader_with_ar():
        try:
            if (df.loc[df["image"] == image].shape[0] > 0):
                continue
            df = df.append({"image": image, "lat": data["latitude"], "long": data["longitude"], "generationtime_ms": data["generationtime_ms"], "utc_offset_seconds": data["utc_offset_seconds"],
                            "timezone": data["timezone"], "elevation": data["elevation"], "time": data["hourly"]["time"], "temperature_2m": data["hourly"]["temperature_2m"], "ar": 1}, ignore_index=True)
        except:
            print("error")
    for data, image in data_loader_with_no_ar():
        try:
            if (df.loc[df["image"] == image].shape[0] > 0):
                continue
            df = df.append({"image": image, "lat": data["latitude"], "long": data["longitude"], "generationtime_ms": data["generationtime_ms"], "utc_offset_seconds": data["utc_offset_seconds"],
                            "timezone": data["timezone"], "elevation": data["elevation"], "time": data["hourly"]["time"], "temperature_2m": data["hourly"]["temperature_2m"], "ar": 0}, ignore_index=True)
        except:
            print("error")
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
#             if year > 2002 and iwv > 30.0:
#                 lines.append({"year": year, "iwv": iwv, "line": line})
#         lines.sort(key=operator.itemgetter("iwv", "year"))
#         for line in lines:
#             file2.write(line["line"])

# for weather_data in data_loader():
#     print(weather_data)
save_data()
# show_data()