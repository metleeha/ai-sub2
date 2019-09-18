# -*- coding: utf-8 -*-
import pickle
from threading import Thread
import sqlite3
from util import extract_features, tokenize
import random
import numpy as np

from flask import Flask
from slack import WebClient
from slackeventsapi import SlackEventAdapter

# Giphy 저장 부분 
import requests
import os
OS_PATH = os.path.dirname(__file__)

def send_request(query):
    api_key = 'GIPHY API KEY'
    base_url = "http://api.giphy.com/v1/gifs/search?q={}&api_key={}&limit=20".format(query, api_key)
    res = requests.get(base_url)
    #url을 대상으로 파일 다운로드를 한다. 
    gifs = res.json().get('data')
    return [ (g.get('images').get('downsized').get('url'), g.get('title')) for g in gifs ]

duo = send_request('you are robot')
win = send_request('winner dance')
lose = send_request('loser')

# slack 연동 정보 입력 부분
SLACK_TOKEN = 'SLACK AUTH TOKEN'
SLACK_SIGNING_SECRET = 'SLACKE SIGNING SECRET'
app = Flask(__name__)

slack_events_adaptor = SlackEventAdapter(SLACK_SIGNING_SECRET, "/listening", app)
slack_web_client = WebClient(token=SLACK_TOKEN)

# Read data
def read_data(filename, filetype):
    if filetype == "txt":
        with open(filename, mode='r', encoding='utf-8') as f:
            data = [line.split('\t') for line in f.read().splitlines()]
    elif filetype == "npy":
        data = np.load(filename)
    elif filetype == "pickle":
        with open(filename, mode='r', encoding='utf-8') as f:
            data = pickle.load(f)
    return data

# Application Initiate function define
def app_init():
    print("Init start")

    if os.path.isfile('ovendata'):
        dataset = np.load('ovendata')
    else:
        word_indices = None
        clf = None
        users_info = {}

        # Req 2-2-1. pickle로 저장된 model.clf 파일 불러오기
        with open('word_indices.pkl', 'rb') as f:
            word_indices = pickle.load(f)
        with open("model1.clf", "rb") as f:
            clf = pickle.load(f)

        dataset = read_data("./Train_data/datafiles/splitLine.txt", "txt")
        X_data = []
        iter = 0
        len_dataset = len(dataset)
        while iter < len_dataset:
            if(len(dataset[iter][0]) == 0 or len(dataset[iter]) < 3 or len(dataset[iter][2]) < 1):
                del(dataset[iter])
                iter -= 1
                len_dataset -= 1
            else:
                dataset[iter][0] = int(dataset[iter][0])
                X_data.append(dataset[iter][2])
            iter += 1

        X_docs = []

        for i in range(len(X_data)):
            X_docs.append(tokenize(X_data[i]))

        # Req 2-2-2. 토큰화 및 one-hot 임베딩하는 전 처리
        embedding = extract_features(X_docs, word_indices)
        # Req 2-2-3. 긍정 혹은 부정으로 분류
        model_pred = clf.predict(embedding)
        for iter in range(len(dataset)):
            dataset[iter].append(int(model_pred[iter]))

        print("Init finish")

        # 정리된 파일 저장하기 
        np.save(OS_PATH + "/ovendata", dataset)

    return clf, word_indices, dataset, users_info

# Application Initiating
clf, word_indices, dataset, users_info = app_init()

# 게임 시작 메소드. 정답 레이블, 그리고 문제를 출력한다.
def start_game(user):
    commentIndex = np.random.randint(0, len(dataset), size=1)[0]
    users_info[user]=commentIndex
    message = "*영화 제목* : "+dataset[commentIndex][1]+"\n *댓글* : "+dataset[commentIndex][2]
    return message

def welcome():
    keyword = {}
    keyword["title"] = "Welcome to Ovengers"
    keyword["text"] = "Welcome Text"
    return keyword

def respond_game(users, text):
    # 유저가 등록되어있는지 확인
    if users in users_info.keys():
        keyword = {}
        idx = users_info[users] # 영화 데이터 인덱스
        naver_rate = dataset[idx][0] # 실제평점
        oven_rate = dataset[idx][3] # 예상평점
        user_rate = int(text)
        # 평점 비교 승부
        num = random.randrange(19)
        keyword["text"] = "*네이버*:{} ".format(naver_rate) + "*오벤져스*:{} ".format(oven_rate) + "*{}*:{}".format(users, user_rate)
        if abs(naver_rate - oven_rate) == abs(naver_rate - user_rate):
            keyword["title"] = "*당신은 진정한 AI인!*"
            keyword["image"] = duo[num]
        elif abs(naver_rate - oven_rate) < abs(naver_rate - user_rate):
            keyword["title"] = "*아이고... AI가 한수위!*"
            keyword["image"] = lose[num]
        else:
            keyword["title"] = "*AI보다 뛰어난 당신!*"
            keyword["image"] = win[num]
        # 유저정보 삭제
        del users_info[users]

        return keyword
    else:
        return welcome()

# Req 2-2-4. app.db 를 연동하여 웹에서 주고받는 데이터를 DB로 저장
def event_handler(event_data):
    keyword = [{}]

    text = event_data["event"]["text"]
    user = event_data["event"]["user"]

    if "시작" in text:
        print("시작 메소드")
        keyword = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Game Start :star: *\n" + start_game(user)
                }
            },
            {
                "type": "divider"
            }
        ]
        # keyword[0]["title"] = "Game Start! \n[Quiz] 평점을 예측해주세요!"
        # keyword[0]["text"] = start_game(user)
    elif "답" in text:
        print("대답 메소드")
        answer = text.split(" ")[-1]
        print(answer)
        if type(answer) != type(2):
            keyword = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "정답 형식을 확인해주세요!" + "\n" + "@유저이름 답 평가점수"
                    }

                }
            ]
        else:
            response= respond_game(user, answer)
            # keyword[0]["title"] = response["title"]
            # keyword[0]["text"] = response["text"]
            # keyword[0]["image_url"] = response["image_url"]
            keyword = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "게임결과"
                    }
                },
                {
                    "type": "divider"
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*Naver:* 점수\n *Ovengers:* 점수 \n *User:* 점수"
                    }
                },
                {
                    "type": "image",
                    "title": {
                        "type": "plain_text",
                        "text": "image1",
                        # "emoji": true
                    },
                    "image_url": "",
                    "alt_text": "image1"
                }
            ]
            keyword[0]["text"]["text"] = response["title"]
            keyword[2]["text"]["text"] = response["text"]
            keyword[3]["image_url"] = response["image"][0]
            keyword[3]["alt_text"] = response["image"][1]
            keyword[3]["title"]["text"] = response["image"][1]
    else:
        print("welcome 메소드")
        keyword = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "안녕 👋 나는 Ovenbot이야. 영화를 좋아하고 댓글 구경하는게 취미야."
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Ovengers Game* :ghost: \n *누가 더 Robot인가?*, ~틀리면 자존심 상할거야~ \n 재밌을거야 :speak_no_evil:"
                }
            },
            {
                "type": "divider"
            },
            {
                "type": "section",
                "text": {
                    "text": "게임을 시작하고 진행하기 위한 가이드야.",
                    "type": "mrkdwn"
                },
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": "*Start*"
                    },
                    {
                        "type": "mrkdwn",
                        "text": "*Answer*"
                    },
                    {
                        "type": "plain_text",
                        "text": "@Username 시작"
                    },
                    {
                        "type": "plain_text",
                        "text": "@Username 답 예상평점"
                    }
                ]
            }
        ]
        # keyword[0]["title"] = "Ovengers Game에 오신것을 환영합니다."
        # keyword[0]["text"] = "게임을 시작하시려면 @챗봇이름 시작 을 입력해주세요.\n정답을 맞출 때는 @챗봇이름 답 점수 이렇게 입력해주세요!"
    return keyword

    # 챗봇이 멘션을 받았을 경우
@slack_events_adaptor.on("app_mention")
def app_mentioned(event_data):
    channel = event_data["event"]["channel"]
    keywords = event_handler(event_data)

    slack_web_client.chat_postMessage(
        channel=channel,
        # attachments=keywords
        blocks=keywords
    )


@app.route("/", methods=["GET"])
def index():
    return "<h1>Server is ready.</h1>"


if __name__ == '__main__':
    app.run(port=8080)
