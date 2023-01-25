import api
import scraper


def data_loader():
    with open("weather_data.txt") as file:
        for line in reversed(file.readlines()):
            numbers = line.split()
            year, month, day, lat, lng = int(numbers[1]), int(numbers[2]), int(
                numbers[3]), float(numbers[4]), float(numbers[5])
            if year < 2001:
                break
            date = f"{year}-{month}-{day}-T00:00:00Z"
            pathname = f"images/image_{year}_{month}_{day}_{lat}_{lng}.png"
            weather_data = api.get_weather_data(year, month, day, lat, lng)
            scraper.save_image(
                lat, lng, zoom=6.5, date=date, pathname=pathname)
            yield weather_data


# just learned generator functions!! yayy
for weather_data in data_loader():
    print(weather_data)
