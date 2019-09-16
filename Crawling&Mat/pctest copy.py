import requests
import time
import numpy as np
from bs4 import BeautifulSoup
import os

file2 = open(os.path.dirname(__file__) + "/comments2.txt", 'w', encoding="utf-8")
file4 = open(os.path.dirname(__file__) + "/delete.txt", 'w', encoding="utf-8")

baseURL = "https://movie.naver.com/movie/point/af/list.nhn?st=mcode&sword="
pageURL = "&page="

movieCode = []

with open(os.path.dirname(__file__) + "/movieID.txt", mode='r', encoding='utf-8') as f:
    movieCode = f.read().splitlines()

iter = 1000    # 크롤링 할 횟수
# sleepSec = 1  # 스레드 쉬는 타임 

np.random.shuffle(movieCode)

for code in movieCode:
    url = baseURL + code
    resp = requests.get(url)
    html = BeautifulSoup(resp.content, 'html.parser')

    total = int(html.find('strong', {'class': 'c_88'}).getText())

    iter = 1000 if total > 10000 else (int(total / 10) + 1)

    pos = 0
    neg = 0

    print(code + " start!")
    file4.write(code + "\n")

    for i in range(iter):
        url = baseURL + code + pageURL + str(i)

        resp = requests.get(url)
        html = BeautifulSoup(resp.content, 'html.parser')

        points = html.findAll('td', {'class': 'point'})
        titles = html.findAll('td', {'class': 'title'})

        for j in range(len(points)):
            point = points[j].getText()
            sprstr = titles[j].getText().split('\n')
            pint = int(point)

            if pint > 5 and pint < 10:
                pos+=1
                file2.write(point + "\t" + sprstr[1] + "\t" + sprstr[2] + "\n")
            elif pint < 5:
                neg+=1
                file2.write(point + "\t" + sprstr[1] + "\t" + sprstr[2] + "\n")

            # sprstr[1] title
            # sprstr[2] comment

    print(code + " : " + str(pos + neg))

file2.close()
file4.close()