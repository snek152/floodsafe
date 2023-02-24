from fastapi import FastAPI
import uvicorn
from utils import get_weather_data
from utils import save_image
import datetime
from utils import predict_data
from uuid import uuid4
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def home(lat: float, long: float):
    date = datetime.datetime.now()
    weather_data = get_weather_data(date.year, date.month, date.day, lat, long)
    date_string = date.strftime("%Y-%m-%d T%H:%M:%SZ")
    pathname = str(uuid4()) + ".png"
    save_image(lat, long, 8, date_string, pathname)
    print(pathname)
    temp, humidity, dewpoint, precip = weather_data["hourly"]["temperature_2m"][date.hour-1], weather_data["hourly"]["relativehumidity_2m"][date.hour -
                                                                                                                                            1], weather_data["hourly"]["dewpoint_2m"][date.hour-1], weather_data["hourly"]["precipitation"][date.hour-1]

    print(temp, humidity, dewpoint, precip)
    data = predict_data(pathname, temp, humidity, dewpoint, precip)
    # os.remove(pathname)
    # return {"ar": predict_data("image.png", 0.0, 2.0, 2.0, 2.0)}
    return {"ar": data}

if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
