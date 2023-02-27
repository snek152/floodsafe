import time
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options


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
    options = Options()
    options.headless = True
    options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(service=Service(
        executable_path=ChromeDriverManager().install()), options=options)
    url = f"https://worldview.earthdata.nasa.gov/?v={long-zoom},{lat-zoom},{long+zoom},{lat+zoom}&l=Reference_Labels_15m(hidden),Reference_Features_15m(hidden),Coastlines_15m(hidden),AMSRU2_Total_Precipitable_Water_Night,AMSRU2_Total_Precipitable_Water_Day,AMSRE_Columnar_Water_Vapor_Day,AMSRU2_Columnar_Water_Vapor_Day,AMSRE_Columnar_Water_Vapor_Night,AMSRU2_Columnar_Water_Vapor_Night&lg=false&t={date}"
    driver.get(url)
    time.sleep(3)
    driver.find_element(By.ID, "toggleIconHolder").click()
    driver.find_element(By.ID, "timeline-hide").click()
    driver.find_element(By.ID, "wv-map").screenshot(pathname)
    crop_image(pathname)
    driver.quit()


if __name__ == "__main__":
    save_image(40, -122, 18, "2019-12-29 07:53:06", "screenshot.png")
