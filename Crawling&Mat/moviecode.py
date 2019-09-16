import requests
import time
from bs4 import BeautifulSoup
import os

file = open(os.path.dirname(__file__) + "/movieID.txt", 'w', encoding="utf-8")

baseURL = "https://movie.naver.com/movie/sdb/rank/rmovie.nhn?sel=pnt&date=20190905&tg="
pageURL = "&page="

tageCodePage = [
    [1, 21],
    [2, 3],
    [4, 5],
    [5, 6],
    [6, 3],
    [7, 3],
    [10, 2],
    [11, 10],
    [13, 2],
    [15, 8],
    [16, 5],
    [18, 3],
    [19, 14]
]

for item in tageCodePage:
    for j in range(item[1]):
        url = baseURL + str(item[0]) + pageURL + str(j)
        resp = requests.get(url)
        html = BeautifulSoup(resp.content, 'html.parser')

        titles = html.findAll('td', {'class': 'title'})

        for title in titles:
            a = title.find('a')
            file.write(a["href"].split("=")[1] + "\n")


file.close()