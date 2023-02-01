import time
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By


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
    driver = webdriver.Chrome(service=Service(
        executable_path=ChromeDriverManager().install()))
    url = f"https://worldview.earthdata.nasa.gov/?v={long-zoom},{lat-zoom},{long+zoom},{lat+zoom}&l=Reference_Labels_15m(hidden),Reference_Features_15m(hidden),Coastlines_15m(hidden),AMSRU2_Total_Precipitable_Water_Night,AMSRU2_Total_Precipitable_Water_Day,AMSRE_Columnar_Water_Vapor_Day,AMSRU2_Columnar_Water_Vapor_Day,AMSRE_Columnar_Water_Vapor_Night,AMSRU2_Columnar_Water_Vapor_Night&lg=false&t={date}"
    # other_url = "https://worldview.earthdata.nasa.gov/?v=-340.5066279931091,-142.6995676849225,236.2386721655414,209.52537981404555&l=Reference_Labels_15m(hidden),Reference_Features_15m(hidden),Coastlines_15m(hidden),AMSRE_Columnar_Water_Vapor_Day,AMSRU2_Columnar_Water_Vapor_Day,AMSRE_Columnar_Water_Vapor_Night,AMSRU2_Columnar_Water_Vapor_Night,AMSRU2_Total_Precipitable_Water_Night,AMSRU2_Total_Precipitable_Water_Day,BlueMarble_NextGeneration(hidden),VIIRS_NOAA20_CorrectedReflectance_TrueColor(hidden),VIIRS_SNPP_CorrectedReflectance_TrueColor(hidden),MODIS_Aqua_CorrectedReflectance_TrueColor(hidden),MODIS_Terra_CorrectedReflectance_TrueColor(hidden)&lg=true&t=2013-02-21-T20%3A00%3A00Z"
    # other_url = "https://worldview.earthdata.nasa.gov/?v=-125.3095952625997,-57.724467407598226,154.42275900064254,113.11127892103254&l=Reference_Labels_15m(hidden),Reference_Features_15m(hidden),Coastlines_15m(hidden),AMSRU2_Columnar_Water_Vapor_Night,AMSRU2_Columnar_Water_Vapor_Day,BlueMarble_NextGeneration(hidden),VIIRS_NOAA20_CorrectedReflectance_TrueColor(hidden),VIIRS_SNPP_CorrectedReflectance_TrueColor(hidden),MODIS_Aqua_CorrectedReflectance_TrueColor(hidden),MODIS_Terra_CorrectedReflectance_TrueColor(hidden)&lg=true&t=2023-01-25-T16%3A51%3A37Z"
    # other_url = "https://worldview.earthdata.nasa.gov/?v=-175.19751452291467,2.0849570200676837,-37.90523544049556,85.93092122387918&z=2&l=Reference_Labels_15m(hidden),Reference_Features_15m(hidden),Coastlines_15m(hidden),MODIS_Terra_Water_Vapor_5km_Night,MODIS_Terra_Water_Vapor_5km_Day,VIIRS_NOAA20_CorrectedReflectance_TrueColor(hidden),VIIRS_SNPP_CorrectedReflectance_TrueColor,MODIS_Aqua_CorrectedReflectance_TrueColor,MODIS_Terra_CorrectedReflectance_TrueColor(hidden)&lg=true&t=2000-11-14-T20%3A53%3A35Z"
    # other_url = "https://worldview.earthdata.nasa.gov/?v=-128.9,-37.3,-128.7,-37.2&l=Reference_Labels_15m(hidden),Reference_Features_15m(hidden),Coastlines_15m,Land_Water_Map,MODIS_Combined_MAIAC_L2G_ColumnWaterVapor,VIIRS_SNPP_CorrectedReflectance_TrueColor(hidden),MODIS_Aqua_CorrectedReflectance_TrueColor(hidden),MODIS_Terra_CorrectedReflectance_TrueColor(hidden)&lg=false&t=2019-12-29-T07:53:06Z"
    driver.get(url)
    time.sleep(5)
    driver.find_element(By.ID, "toggleIconHolder").click()
    driver.find_element(By.ID, "timeline-hide").click()
    driver.find_element(By.ID, "wv-map").screenshot(pathname)
    crop_image(pathname)
    driver.quit()
    # driver.save_screenshot("screenshot.png")


if __name__ == "__main__":
    save_image(40, -122, 18, "2019-12-29 07:53:06", "screenshot.png")
