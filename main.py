import cv2
import numpy as np

before = cv2.imread('1.png')
after = cv2.imread('2.png')

H = before.shape[0]
W = before.shape[1]

used = []
maxSize = 1e3

for i in range(0, H):
    used.append([])
    for j in range(0, W):
        used[i].append(False)

def bfs(y, x, maxSize):
    global H, W, before, after, used
    leftAns = 1e9
    rightAns = -1e9
    upperAns = 1e9
    bottomAns = -1e9
    col = 0

    q = []

    q.append((y, x))
    while(len(q) != 0):
        leftAns = min(q[0][1], leftAns)
        rightAns = max(q[0][1], rightAns)
        upperAns = min(q[0][0], upperAns)
        bottomAns = max(q[0][0], bottomAns)

        if x + 1 < W and before[q[0][0]][q[0][1] + 1] == after[q[0][0]][q[0][1] + 1] and used[q[0][0]][q[0][1] + 1] == False:
            q.append((q[0][0], q[0][1] + 1))
            col+= 1
        if x - 1 >= 0 and before[q[0][0]][q[0][1] - 1] == after[q[0][0]][q[0][1] - 1] and used[q[0][0]][q[0][1] - 1] == False:
            q.append((q[0][0], q[0][1] - 1))
            col+= 1
        if y + 1 < H and before[q[0][0] + 1][q[0][1]] == after[q[0][0] + 1][q[0][1]] and used[q[0][0] + 1][q[0][1]] == False:
            q.append((q[0][0] + 1, q[0][1]))
            col+= 1
        if y - 1 >= 0 and before[q[0][0] - 1][q[0][1]] == after[q[0][0] - 1][q[0][1]] and used[q[0][0] - 1][q[0][1]] == False:
            q.append((q[0][0] - 1, q[0][1]))
            col+= 1

        q.pop()

    if col > maxSize:
        return None
    return (leftAns, rightAns, upperAns, bottomAns)

for i in range(0, H):
    for j in range(0, W):
        leftAns, rightAns, upperAns, bottomAns = bfs(i, j, maxSize)

        sample = before[upperAns:bottomAns, leftAns:rightAns]
        


cv2.waitKey(0)
