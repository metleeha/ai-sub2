# 오벤져스 AI 2~3주차

## Slack Chat Bot
![웰컴](https://user-images.githubusercontent.com/27988544/64404330-8675c080-d0b6-11e9-9b9a-131a8f05f77b.PNG)

![봇시작](https://user-images.githubusercontent.com/27988544/64404367-a7d6ac80-d0b6-11e9-996f-f127279b6eb4.PNG)

![봇답](https://user-images.githubusercontent.com/27988544/64404392-c0df5d80-d0b6-11e9-8ed9-b14b750ad8ce.PNG)

> @봇이름 시작 : 게임을 시작하는 명령
>
> @봇이름 답 '숫자' : 문제에 대한 답을 입력, 결과를 받기 위한 명령어
>
> @봇이름 (그 외) : Welcome 메세지 출력 (사용방법)

- <b>기계인 측정 봇</b>
- 영화 제목과 평가 내용이 질문으로 주어지면, 사용자가 정답을 맞추도록 유도하는 챗봇
- 실제 평점에 누가 더 가까운지를 판별해서 대답 출력 (머신러닝 모델 VS 유저)

## Data Distribution Graph

![Data_Distribution](https://user-images.githubusercontent.com/27988544/64401557-62ad7d00-d0ac-11e9-82e1-175da5148ac2.png)

- 네이버 영화 - 모든 영화에 대해 크롤링 진행, 각 클래스별 데이터 수의 균형이 최대한 유지될 수 있도록 작업 하였음.

## AWS
AWS의 EC2를 챗봇 서버로 이용하였음.

## CI/CD and Docker
그리고 Git과 Docker를 연동해서 배포 버전이 올라갈 경우 자동으로 서버에서 갱신될 수 있도록 세팅하였음.