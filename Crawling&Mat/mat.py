import matplotlib.pyplot as plt
import os

fig = plt.figure()
plt.title('Data Label Visualization')

data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
with open(os.path.dirname(__file__) + "/commentsMain.txt", mode='r', encoding='utf-8') as f:
    for line in f.read().splitlines():
        curLine = line.split('\t')
        if len(curLine) == 3:
            curData = curLine[0]
            if len(curData) > 0:
                data[int(curData) -1] += 1

plt.bar(range(1, 11), data)
plt.show()