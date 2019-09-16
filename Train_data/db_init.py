# -*- coding: utf-8 -*-
from urllib.request import FancyURLopener
from bs4 import BeautifulSoup
import sqlite3
conn = sqlite3.connect('app.db')

c = conn.cursor()

# CREATE DATABASE FILE
# c.execute('''CREATE TABLE search_history
#              (query text,
#              rating number)''')

# Req
def insertData(rating, title, comment):
    c.execute("INSERT INTO search_history VALUES (?, ?, ?)", (rating, title, comment))

# sprstr[1] title
# sprstr[2] comment
# c.executemany("INSERT INTO search_history VALUES (?, ?)", inputData)

conn.commit()
conn.close()