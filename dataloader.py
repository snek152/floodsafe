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
from PIL import Image
import glob


def data_loader_with_ar(df):
    with open("processed_weather_data.txt") as file:
        for line in file:
            numbers = line.split()
            year, month, day, hour, lat, lng, iwv = int(numbers[1]), int(numbers[2]), int(
                numbers[3]), int(numbers[4]), float(numbers[5]), float(numbers[6])-360, float(numbers[10])
            date = datetime.datetime(year, month, day, hour)
            date_string = date.strftime("%Y-%m-%d T%H:%M:%SZ")
            pathname = f"images/image_{year}_{month}_{day}_{hour}_{lat}_{lng}.png"
            print(lat, lng, iwv)

            if os.path.exists(pathname) and df.loc[df["image"] == pathname].shape[0] <= 0:
                print("file exists")
            else:
                continue
                # scraper.save_image(
                #     lat, lng, zoom=8, date=date_string, pathname=pathname)
            weather_data = api.get_weather_data(year, month, day, lat, lng)
            yield weather_data, pathname, hour


def data_loader_with_no_ar():
    # with open("processed_weather_data.txt") as file:
    while len(os.listdir("no_ar_images")) < 1156:
        lat, long = tuple(zip(np.random.uniform(-90., 90., 1),
                              np.random.uniform(-180., 180., 1)))[0]
        year = np.random.randint(2002, 2023)
        month = np.random.randint(1, 13)
        day = np.random.randint(1, 29)
        hour = np.random.randint(0, 24)
        date = datetime.datetime(year, month, day, hour)

        date_string = date.strftime("%Y-%m-%d T%H:%M:%SZ")
        pathname = f"no_ar_images/image_{year}_{month}_{day}_{hour}_{lat}_{long}.png"
        scraper.save_image(lat, long, zoom=8,
                           date=date_string, pathname=pathname)
        im = set(
            list(map(lambda x: x[0], list(Image.open(pathname).convert("HSV").getdata()))))
        if (set(list(Image.open(pathname).getdata())) == {(0, 0, 0, 255)} or len(im.difference({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 170, 171, 172, 173, 174, 175, 175, 176, 177, 178, 179, 180})) != len(im)):
            os.remove(pathname)
            continue
        weather_data = api.get_weather_data(year, month, day, lat, long)
        yield weather_data, pathname, hour


def save_data():
    # df = pd.DataFrame(
    #     columns=["image", "lat", "long", "generationtime_ms", "utc_offset_seconds", "timezone", "elevation", "time", "temperature_2m", "ar", "hour"])
    df = pd.read_pickle("data.pkl")
    # try:
    #     for data, image, hour in data_loader_with_ar(df):
    #         # if (df.loc[df["image"] == image].shape[0] > 0):
    #         #     continue
    #         df = df.append({"image": image, "lat": data["latitude"], "long": data["longitude"], "generationtime_ms": data["generationtime_ms"], "utc_offset_seconds": data["utc_offset_seconds"],
    #                         "timezone": data["timezone"], "elevation": data["elevation"], "time": data["hourly"]["time"][hour-1], "temperature_2m": data["hourly"]["temperature_2m"][hour-1], "ar": 1, "hour": hour}, ignore_index=True)
    # except Exception as e:
    #     print(e)
    try:
        for data, image, hour in data_loader_with_no_ar():
            df = df.append({"image": image, "lat": data["latitude"], "long": data["longitude"], "generationtime_ms": data["generationtime_ms"], "utc_offset_seconds": data["utc_offset_seconds"],
                            "timezone": data["timezone"], "elevation": data["elevation"], "time": data["hourly"]["time"][hour-1], "temperature_2m": data["hourly"]["temperature_2m"][hour-1], "ar": 0, "hour": hour}, ignore_index=True)
    except Exception as e:
        print(e)
    df.to_pickle("data.pkl")


def show_data():
    df = pd.read_pickle("data.pkl")
    print(len(df))
    print(df.tail())
    print(len(df[df["ar"] == 1]))
    print(len(df[df["ar"] == 0]))
    print("No Images length", len(os.listdir("no_ar_images/")))
    print("Images length", len(os.listdir("images/")))
# df = pd.read_csv("data.csv")
# for index, row in df.iterrows():
#     image = cv2.imread(row["image"])
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     plt.imshow(image)
#     plt.show()


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
# df = pd.read_csv("data.csv")
# df = df[df["ar"] == 1]
# df = df.drop(columns=["Unnamed: 0"])
# df.to_csv("data.csv")
# df.to_csv("data.csv")
# show_data()
