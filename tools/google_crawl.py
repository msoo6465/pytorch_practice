import json
import os
import time

import urllib3
from selenium import webdriver

options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('window-size=1920x1080')
options.add_argument("disable-gpu")

searchterm = '강아지'  # will also be the name of the folder

if __name__ == '__main__':
    # NEED TO DOWNLOAD CHROMEDRIVER, insert path to chromedriver inside parentheses in following line(https://sites.google.com/a/chromium.org/chromedriver/downloads)
    browser = webdriver.Chrome(os.path.expanduser('~/Downloads/chromedriver'),chrome_options=options)

    url = "https://www.google.co.in/search?q=" + searchterm + "&source=lnms&tbm=isch"
    browser.get(url)

    counter = 0
    succounter = 0

    result_dir = '~/Downloads/image_crawling/google'
    result_dir = os.path.expanduser(result_dir)
    result_dir = os.path.join(result_dir, searchterm)

    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    for _ in range(500):
        browser.execute_script("window.scrollBy(0,10000)")

        isShowBottom = browser.find_element_by_class_name("DwpMZe ").get_attribute('data-status') # 마지막 바닥 요소
        isShowMoreButton = browser.find_element_by_class_name('mye4qd').is_displayed()  # 버튼 요소

        if isShowMoreButton:
            time.sleep(1)
            browser.find_element_by_class_name('mye4qd').click()

        if isShowBottom==3 and not isShowMoreButton:
            # 마지막 바닥과 버튼 요소가 보이지 않으므로 검색이 끝난것으로 판단하고 종료
            break

    retries = urllib3.Retry(connect=5, read=2, redirect=5)
    http = urllib3.PoolManager(retries=retries)
    link = []
    for idx,x in enumerate(browser.find_elements_by_class_name('rg_i')):
        counter = counter + 1
        try:
            os.makedirs('tmp',exist_ok=True)
            link.append(x.get_attribute('src'))
            succounter = succounter + 1
        except Exception as e:
            print("can't get img")
            print(e)
    print(link)
    import urllib.request

    count = 0
    for url in link:
        try:
            count+=1
            urllib.request.urlretrieve(url,'./tmp/img'+str(count)+'.jpg')
        except Exception as e:
            print('Error',e)
    print(count, "pictures succesfully downloaded")
    browser.close()
