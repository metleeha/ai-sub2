import pickle
from threading import Thread
import sqlite3
from util import extract_features, tokenize
import random
import numpy as np

from flask import Flask
from slack import WebClient
from slackeventsapi import SlackEventAdapter

with open("model1.clf", "rb") as f:
    clf = pickle.load(f)
with open('word_indices.pkl', 'rb') as f:
    word_indices = pickle.load(f)

while True:
    str1 = input()

    if str1 == "exit":
        break
    if str1 == "그만":
        break

    sliceStr = [tokenize(str1)]
    print(sliceStr)

    X_test = extract_features(sliceStr, word_indices)

    print(X_test)

    pred = clf.predict(X_test)

    print(pred)

