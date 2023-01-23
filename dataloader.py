import api
import scraper



def data_loader():
    with open("weather_data.txt") as file:
        for line in file:
            numbers = line.split()
            year, month, day, lat, lng = int(numbers[1]), int(numbers[2]), int(numbers[3]), float(numbers[4]), float(numbers[5])
            date = f"{year}-{month}-{day}-T00:00:00Z"
            pathname = f"image_{year}_{month}_{day}_{lat}_{lng}.png"
            weather_data = api.get_weather_data(year, month, day, lat, lng)
            image = scraper.save_image(lat, lng, zoom=3, date=date, pathname=pathname)
            yield weather_data, image

# just learned generator functions!! yayy



