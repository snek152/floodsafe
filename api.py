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


download_image()
# print(get_data())
