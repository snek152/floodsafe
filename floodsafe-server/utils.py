import time
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import requests
import json
import csv
import datetime

import pandas as pd
import torchvision
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import time
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch


class MyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, device):
        self.images = []
        self.weather_data = []
        self.labels = []
        self.device = device
        for i, row in df.iterrows():
            image = Image.open(row['image'])
            image = image.convert('RGB')
            self.images.append(image)
            self.weather_data.append(np.array(
                [row['temperature'], row["humidity"], row["dewpoint"], row["precipitation"]]))
            self.labels.append(row['ar'])
        # for i, filename in enumerate(os.listdir(folder_path)):
        #     if filename.endswith('.png'):
        #         image_path = os.path.join(folder_path, filename)
        #         image = Image.open(image_path)

        #         image = image.convert('RGB')
        #         self.images.append(image)
        #         label = 0 if i < n else 1
        #         self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        tensor_image = transforms.ToTensor()(image)
        weather_data = self.weather_data[index]
        # weather_data.reshape(-1, 1)
        # tensor_data = transforms.ToTensor()(weather_data)
        tensor_data = weather_data
        label = self.labels[index]
        return tensor_image.to(self.device), tensor_data, label

    def collate_fn(self, batch):
        images = [item[0] for item in batch]
        weather_data = [item[1] for item in batch]
        target = [item[2] for item in batch]
        images = torch.stack(images, dim=0)
        weather_data = torch.tensor(
            np.array(weather_data), dtype=torch.float)
        target = torch.tensor(target, dtype=torch.float)
        return images, weather_data, target


class Model(nn.Module):
    def __init__(self, num_classes, num_weather_features):
        super(Model, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.fc1 = nn.Linear(1000 + num_weather_features, 512)
        # self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(512, num_classes)
        # self.resnet.fc = nn.Linear(512, num_classes)

        # Add an additional linear layer for the weather data
        # self.weather_fc = nn.Linear(num_weather_features, 64)

    def forward(self, x, weather_data):
        x = self.resnet(x)
        # add two dimensions to match the ResNet output tensor
        # weather_data = weather_data.unsqueeze(-1)
        # repeat the weather data tensor to match the ResNet output tensor
        # weather_data = weather_data.repeat(1, 1)
        x = x.view(x.size(0), -1)
        # returning torch.Size([64, 1000]) torch.Size([64, 1]) PLS HELP
        x = torch.cat((x, weather_data), dim=1)
        x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        x = self.fc3(x)
        # x = self.resnet.bn1(x)
        # x = self.resnet.relu(x)
        # x = self.resnet.maxpool(x)

        # x = self.resnet.layer1(x)
        # x = self.resnet.layer2(x)
        # x = self.resnet.layer3(x)
        # x = self.resnet.layer4(x)

        # x = F.avg_pool2d(x, x.shape[2])
        # x = x.view(x.size(0), -1)

        # # Concatenate the weather data to the output of the convolutional layers
        # weather_data = F.relu(self.weather_fc(weather_data))
        # x = torch.cat((x, weather_data), dim=1)

        # x = self.resnet.fc(x)
        return x


def get_weather_data(year: int, month: int, day: int, lat: float, lng: float):
    start_date = datetime.datetime(
        year, month, day) - datetime.timedelta(days=1)
    end_date = datetime.datetime(year, month, day)
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&start_date={start_date.strftime('%Y-%m-%d')}&end_date={end_date.strftime('%Y-%m-%d')}&hourly=temperature_2m,relativehumidity_2m,dewpoint_2m,precipitation"
    response = requests.get(url)
    # print(type(response.json()))
    return response.json()


def crop_image(pathname: str):
    im = Image.open(pathname)
    width, height = im.size
    crop_radius = 300
    left = width//2 - crop_radius
    right = width//2 + crop_radius
    top = height//2 - crop_radius
    bottom = height//2 + crop_radius
    cropped_im = im.crop((left, top, right, bottom))
    cropped_im.save(pathname)


def save_image(lat, long, zoom, date, pathname):
    options = Options()
    options.headless = True
    options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(service=Service(
        executable_path=ChromeDriverManager().install()), options=options)
    url = f"https://worldview.earthdata.nasa.gov/?v={long-zoom},{lat-zoom},{long+zoom},{lat+zoom}&l=Reference_Labels_15m(hidden),Reference_Features_15m(hidden),Coastlines_15m(hidden),AMSRU2_Total_Precipitable_Water_Night,AMSRU2_Total_Precipitable_Water_Day,AMSRE_Columnar_Water_Vapor_Day,AMSRU2_Columnar_Water_Vapor_Day,AMSRE_Columnar_Water_Vapor_Night,AMSRU2_Columnar_Water_Vapor_Night&lg=false&t={date}"
    # other_url = "https://worldview.earthdata.nasa.gov/?v=-340.5066279931091,-142.6995676849225,236.2386721655414,209.52537981404555&l=Reference_Labels_15m(hidden),Reference_Features_15m(hidden),Coastlines_15m(hidden),AMSRE_Columnar_Water_Vapor_Day,AMSRU2_Columnar_Water_Vapor_Day,AMSRE_Columnar_Water_Vapor_Night,AMSRU2_Columnar_Water_Vapor_Night,AMSRU2_Total_Precipitable_Water_Night,AMSRU2_Total_Precipitable_Water_Day,BlueMarble_NextGeneration(hidden),VIIRS_NOAA20_CorrectedReflectance_TrueColor(hidden),VIIRS_SNPP_CorrectedReflectance_TrueColor(hidden),MODIS_Aqua_CorrectedReflectance_TrueColor(hidden),MODIS_Terra_CorrectedReflectance_TrueColor(hidden)&lg=true&t=2013-02-21-T20%3A00%3A00Z"
    # other_url = "https://worldview.earthdata.nasa.gov/?v=-125.3095952625997,-57.724467407598226,154.42275900064254,113.11127892103254&l=Reference_Labels_15m(hidden),Reference_Features_15m(hidden),Coastlines_15m(hidden),AMSRU2_Columnar_Water_Vapor_Night,AMSRU2_Columnar_Water_Vapor_Day,BlueMarble_NextGeneration(hidden),VIIRS_NOAA20_CorrectedReflectance_TrueColor(hidden),VIIRS_SNPP_CorrectedReflectance_TrueColor(hidden),MODIS_Aqua_CorrectedReflectance_TrueColor(hidden),MODIS_Terra_CorrectedReflectance_TrueColor(hidden)&lg=true&t=2023-01-25-T16%3A51%3A37Z"
    # other_url = "https://worldview.earthdata.nasa.gov/?v=-175.19751452291467,2.0849570200676837,-37.90523544049556,85.93092122387918&z=2&l=Reference_Labels_15m(hidden),Reference_Features_15m(hidden),Coastlines_15m(hidden),MODIS_Terra_Water_Vapor_5km_Night,MODIS_Terra_Water_Vapor_5km_Day,VIIRS_NOAA20_CorrectedReflectance_TrueColor(hidden),VIIRS_SNPP_CorrectedReflectance_TrueColor,MODIS_Aqua_CorrectedReflectance_TrueColor,MODIS_Terra_CorrectedReflectance_TrueColor(hidden)&lg=true&t=2000-11-14-T20%3A53%3A35Z"
    # other_url = "https://worldview.earthdata.nasa.gov/?v=-128.9,-37.3,-128.7,-37.2&l=Reference_Labels_15m(hidden),Reference_Features_15m(hidden),Coastlines_15m,Land_Water_Map,MODIS_Combined_MAIAC_L2G_ColumnWaterVapor,VIIRS_SNPP_CorrectedReflectance_TrueColor(hidden),MODIS_Aqua_CorrectedReflectance_TrueColor(hidden),MODIS_Terra_CorrectedReflectance_TrueColor(hidden)&lg=false&t=2019-12-29-T07:53:06Z"
    driver.get(url)
    time.sleep(3)
    driver.find_element(By.ID, "toggleIconHolder").click()
    driver.find_element(By.ID, "timeline-hide").click()
    driver.find_element(By.ID, "wv-map").screenshot(pathname)
    crop_image(pathname)
    driver.quit()
    # driver.save_screenshot("screenshot.png")


@torch.no_grad()
def predict_data(image_path: str, temp: float, humidity: float, dewpoint: float, precipitation: float):
    my_model = Model(2, 4)
    my_model.load_state_dict(torch.load("best_cnn.pt"))
    df = pd.DataFrame(columns=["image", "temperature",
                               "humidity", "dewpoint", "precipitation", "ar"])
    df = df.append({"image": image_path, "temperature": temp, "humidity": humidity,
                   "dewpoint": dewpoint, "precipitation": precipitation, "ar": 0}, ignore_index=True)
    dataset = MyDataset(df, device=torch.device("cpu"))
    loader = DataLoader(dataset, batch_size=1,
                        shuffle=False, num_workers=0, collate_fn=dataset.collate_fn)
    labels = [False, True]
    for data in loader:
        inp, weather_data, _ = data
        outputs = my_model(inp, weather_data)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted
        return labels[predicted[0]]
