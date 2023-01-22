import time
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By


def crop_image(pathname):
    im = Image.open(pathname)
    width, height = im.size
    crop_radius = 200
    left = width/2 - crop_radius
    right = width/2 + crop_radius
    top = height/2 - crop_radius
    bottom = height/2 + crop_radius
    cropped_im = im.crop((left, top, right, bottom))
    cropped_im.save(pathname)


def save_image(lat, long, zoom, date, pathname):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    url = f"https://worldview.earthdata.nasa.gov/?v={long-3},{lat-3},{long+3},{lat+3}&l=Reference_Labels_15m(hidden),Reference_Features_15m(hidden),Coastlines_15m,Land_Water_Map,MODIS_Combined_MAIAC_L2G_ColumnWaterVapor,VIIRS_SNPP_CorrectedReflectance_TrueColor(hidden),MODIS_Aqua_CorrectedReflectance_TrueColor(hidden),MODIS_Terra_CorrectedReflectance_TrueColor(hidden)&lg=false&t={date}"
    # other_url = "https://worldview.earthdata.nasa.gov/?v=-128.9,-37.3,-128.7,-37.2&l=Reference_Labels_15m(hidden),Reference_Features_15m(hidden),Coastlines_15m,Land_Water_Map,MODIS_Combined_MAIAC_L2G_ColumnWaterVapor,VIIRS_SNPP_CorrectedReflectance_TrueColor(hidden),MODIS_Aqua_CorrectedReflectance_TrueColor(hidden),MODIS_Terra_CorrectedReflectance_TrueColor(hidden)&lg=false&t=2019-12-29-T07:53:06Z"
    driver.get(url)
    time.sleep(5)
    driver.find_element(By.ID, "toggleIconHolder").click()
    driver.find_element(By.ID, "timeline-hide").click()
    driver.find_element(By.ID, "wv-map").screenshot(pathname)
    crop_image(pathname)
    # driver.save_screenshot("screenshot.png")


save_image(37, -122, 18, "2019-12-29 07:53:06", "screenshot.png")
