import requests
import json
import os
import csv
import datetime

def get_weather_data(year: int, month: int, day: int, lat: float, lng: float):
    start_date = datetime.datetime(year, month, day)
    end_date = start_date + datetime.timedelta(days=1)
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lng}&start_date={start_date.strftime('%Y-%m-%d')}&end_date={end_date.strftime('%Y-%m-%d')}&hourly=temperature_2m,relativehumidity_2m,dewpoint_2m,precipitation"
    response = requests.get(url)
    return response.json()
