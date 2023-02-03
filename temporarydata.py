from scraper import save_image
from api import get_weather_data
import datetime
import operator
import pandas as pd
import cv2
import os
import glob
# from typings import APIResponse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# to upload images only


def data_loader_with_ar():
    lineCount = 0
    with open("processed_weather_data.txt") as file:
        for line in file:
            lineCount += 1
            if lineCount >= 10:
                break
            numbers = line.split()
            year, month, day, hour, lat, lng, iwv = int(numbers[1]), int(numbers[2]), int(
                numbers[3]), int(numbers[4]), float(numbers[5]), float(numbers[6])-360, float(numbers[10])
            if year < 2001:
                break
            date = datetime.datetime(year, month, day, hour)
            date_string = date.strftime("%Y-%m-%d T%H:%M:%SZ")
            pathname = f"images/image_{year}_{month}_{day}_{hour}_{lat}_{lng}.png"

#            weather_data = get_weather_data(year, month, day, lat, lng)
            save_image(
                lat, lng, zoom=8, date=date_string, pathname=pathname)
            print(lineCount)
            # yield weather_data, pathname, iwv


def cleanup_images():
    # for filename in glob.glob('/Users/snehilk/Desktop/Coding/floodsafe/images/*.png'):
    #     im = Image.open(filename)
    #     pix = list(im.getdata())
    #     if set(pix) == {(0, 0, 0, 255)}:
    #         os.remove(filename)
    #     print('done')
    df = pd.read_pickle("data.pkl")
    for filename in glob.glob('/Users/snehilk/Desktop/Coding/floodsafe/images/*.png'):
        print(df.loc(df["image"] == filename).shape[0])
        if df.loc[df["image"] == filename].shape[0] <= 0:
            os.remove(filename)
            print("removed")
        print('done')


cleanup_images()
