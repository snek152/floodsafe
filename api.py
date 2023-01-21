import requests
import json
import dotenv
import os

dotenv.load_dotenv(".env")


def download_image():
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={37.3013893},{-122.0070246}&key={os.environ.get('API_KEY')}&size={600}x{600}&zoom={18}&maptype=satellite"
    response = requests.get(url)
    with open("image.png", "wb") as f:
        f.write(response.content)


def get_weather_data():
    url = " https://api.ambeedata.com/weather/history/by-lat-lng?lat=12&lng=77&from=2023-01-20 00:00:00&to=2023-01-21 00:00:00"
    response = requests.get(url, headers={
        "x-api-key": os.environ.get("RAPID_API_KEY"),
        "Content-Type": "application/json"
    })

    return response.json()


print(get_weather_data())
# download_image()
# print(get_data())
