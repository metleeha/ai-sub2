import requests
from random import randrange

def send_request(query):
    api_key = '6QhKRkIk8r6rh3O6UH9IjeAsHV7IU1L8'
    base_url = "http://api.giphy.com/v1/gifs/search?q={}&api_key={}&limit=11".format(query, api_key)
    res = requests.get(base_url)
    #url을 대상으로 파일 다운로드를 한다. 
    gifs = res.json().get('data')
    return [(g.get('images').get('downsized').get('url'), g.get('title')) for g in gifs]

number = randrange(10)

if __name__ == "__main__":
    query = input()
    print(send_request(query))