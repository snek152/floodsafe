import api
import scraper
import datetime
import operator


def data_loader():
    with open("processed_weather_data.txt") as file:
        for line in reversed(file.readlines()):
            numbers = line.split()
            year, month, day, hour, lat, lng, iwv = int(numbers[1]), int(numbers[2]), int(
                numbers[3]), int(numbers[4]), float(numbers[5]), float(numbers[6])-360, float(numbers[10])
            if year < 2001:
                break
            date = datetime.datetime(year, month, day)
            # date_string = date.strftime("%Y-%m-%d T%H:%M:%SZ")
            pathname = f"images/image_{year}_{month}_{day}_{lat}_{lng}.png"
            print(lat, lng, iwv)
            weather_data = api.get_weather_data(year, month, day, lat, lng)
            scraper.save_image(
                lat, lng, zoom=6.5, date=date, pathname=pathname)
            yield weather_data


# with open("weather_data.txt") as file:
#     with open("processed_weather_data.txt", "w") as file2:
#         lines = []
#         for line in file:
#             numbers = line.split()
#             year, iwv = int(numbers[1]), float(numbers[10])
#             if year > 2002:
#                 lines.append({"year": year, "iwv": iwv, "line": line})
#         lines.sort(key=operator.itemgetter("iwv", "year"))
#         for line in lines:
#             file2.write(line["line"])
# just learned generator functions!! yayy
for weather_data in data_loader():
    print(weather_data)
