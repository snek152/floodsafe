from fastapi import FastAPI
import uvicorn
import torch
import os
from api import get_weather_data
from scraper import save_image
import datetime
from model import predict_data

app = FastAPI()


@app.get("/")
# lat: float, long: float, year: int, month: int, day: int, hour: int):
async def home():
    # this is all boilerplate btw none of this is gonna work
    # model = torch.load("model.pkl")
    # weather_data = get_weather_data(year, month, day, lat, long)
    # date = datetime.datetime(year, month, day, hour)
    # date_string = date.strftime("%Y-%m-%d T%H:%M:%SZ")
    # image = save_image(lat, long, 8, date_string, "image.png")
    # model predict using weather data and image (will figure out how to do this after u build model so we can use the right parameters and stuff)
    return {"ar": predict_data("images/image_2003_1_8_6_27.5_-112.5.png", 0.0, 2.0, 2.0, 2.0)}

if __name__ == "__main__":
    uvicorn.run(app, reload=True)
