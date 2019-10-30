import cv2
import numpy as np

before = cv2.imread('1.png')
after = cv2.imread('2.png')

H = before.shape[0]
W = before.shape[1]

def bfs(y, x, maxSize):
    global H, W, before, after
    leftAns = 1e9
    rightAns = -1e9
    upperAns = 1e9
    bottomAns = -1e9

    q = []
    used = []
    for i in range(0, H):
        used.append([])
        for j in range(0, W):
            used[i].append(False)

    q.append((y, x))
    while(len(q) != 0):
        if x + 1 < W and before[q[0][0]][q[0][1] + 1] == after[q[0][0]][q[0][1] + 1]:
            q.append((q[0][0], q[0][1]))


cv2.waitKey(0)
