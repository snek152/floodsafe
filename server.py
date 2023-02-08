from fastapi import FastAPI
import uvicorn
import torch
import os
from api import get_weather_data
from scraper import save_image
import datetime

app = FastAPI()


@app.get("/")
async def home(lat: float, long: float, year: int, month: int, day: int, hour: int):
    # this is all boilerplate btw none of this is gonna work
    model = torch.load("model.pkl")
    weather_data = get_weather_data(year, month, day, lat, long)
    date = datetime.datetime(year, month, day, hour)
    date_string = date.strftime("%Y-%m-%d T%H:%M:%SZ")
    image = save_image(lat, long, 8, date_string, "image.png")
    # model predict using weather data and image (will figure out how to do this after u build model so we can use the right parameters and stuff)
    return {"message": "Hello World"}
