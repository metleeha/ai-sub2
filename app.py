import pickle
from threading import Thread
import sqlite3
from util import extract_features, tokenize
import random

import numpy as np
from flask import Flask
from slack import WebClient
from slackeventsapi import SlackEventAdapter
from scipy.sparse import lil_matrix

# slack 연동 정보 입력 부분
SLACK_TOKEN = "xoxb-660478411538-730029623540-fLV7v32hW6LHkwPoVa92ed2k"
SLACK_SIGNING_SECRET = "100c0a308e485a2447decb5ed4a838f6"

app = Flask(__name__)

slack_events_adaptor = SlackEventAdapter(SLACK_SIGNING_SECRET, "/listening", app)
slack_web_client = WebClient(token=SLACK_TOKEN)

# Req 2-2-1. pickle로 저장된 model.clf 파일 불러오기
word_indices = None
with open('word_indices.pkl', 'rb') as f:
    word_indices = pickle.load(f)
clf = None
with open("./model1.clf", "rb") as f:
    clf = pickle.load(f)


# Req 2-2-2. 토큰화 및 one-hot 임베딩하는 전 처리
def preprocess(X):
    return extract_features(X, word_indices)


# Req 2-2-3. 긍정 혹은 부정으로 분류
def classify(X):
    return clf.predict(preprocess(X))


# Req 2-2-4. app.db 를 연동하여 웹에서 주고받는 데이터를 DB로 저장


# 챗봇이 멘션을 받았을 경우
@slack_events_adaptor.on("app_mention")
def app_mentioned(event_data):
    channel = event_data["event"]["channel"]
    text = event_data["event"]["text"]
    inputStr = " ".join(text.split(" ")[1:])
    inputDoc = [[tokenize(inputStr)]]
    result = classify(inputDoc)
    if result == 1:
        result = "긍정적 리뷰"
    else:
        result = "부정적 리뷰"

    R = str(hex(random.randrange(0, 256)))[2:]
    G = str(hex(random.randrange(0, 256)))[2:]
    B = str(hex(random.randrange(0, 256)))[2:]
    keyword = [{}]
    keyword[0]["color"] = "#" + R + G + B
    keyword[0]["text"] = "결과는 " + result
    keyword[0]["title"] = "영화 예상 평가"

    slack_web_client.chat_postMessage(
        channel=channel,
        attachments=keyword
    )


@app.route("/", methods=["GET"])
def index():
    return "<h1>Server is ready.</h1>"


if __name__ == '__main__':
    app.run(port=8080)
