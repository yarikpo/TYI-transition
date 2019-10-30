import cv2
import numpy as np

before = cv2.imread('1.png')
after = cv2.imread('2.png')

H = before.shape[0]
W = before.shape[1]

# W, H = before.shape[::-1]

used = []
maxSize = H * W - 1e5

for i in range(0, H):
    used.append([])
    for j in range(0, W):
        used[i].append(False)

def check(y1, x1, y2, x2):
    global before, after
    if before[y1][x1][0] != before[y2][x2][0] or before[y1][x1][1] != before[y2][x2][1] or before[y1][x1][2] != before[y2][x2][2]:
        return False
    return True

def bfs(y, x, maxSize):
    print('bfs!')
    global H, W, before, after, used
    leftAns = 1e8
    rightAns = -1e8
    upperAns = 1e8
    bottomAns = -1e8
    col = 0

    q = []

    q.append((y, x))
    while(len(q) != 0):
        # used[q[0][0]][q[0][1]] = True
        if col % 10000 == 0:
            print(col)
        leftAns = min(q[0][1], leftAns)
        rightAns = max(q[0][1], rightAns)
        upperAns = min(q[0][0], upperAns)
        bottomAns = max(q[0][0], bottomAns)

        if q[0][1] + 1 < W and check(q[0][0], q[0][1], q[0][0], q[0][1] + 1) == True and used[q[0][0]][q[0][1] + 1] == False:
            used[q[0][0]][q[0][1] + 1] = True
            q.append((q[0][0], q[0][1] + 1))
            col+= 1
        if q[0][1] - 1 >= 0 and check(q[0][0], q[0][1], q[0][0], q[0][1] - 1) == True and used[q[0][0]][q[0][1] - 1] == False:
            used[q[0][0]][q[0][1] - 1] = True
            q.append((q[0][0], q[0][1] - 1))
            col+= 1
        if q[0][0] + 1 < H and check(q[0][0], q[0][1], q[0][0] + 1, q[0][1]) == True and used[q[0][0] + 1][q[0][1]] == False:
            used[q[0][0] + 1][q[0][1]] = True
            q.append((q[0][0] + 1, q[0][1]))
            col+= 1
        if q[0][0] - 1 >= 0 and check(q[0][0], q[0][1], q[0][0] - 1, q[0][1]) == True and used[q[0][0] - 1][q[0][1]] == False:
            used[q[0][0] - 1][q[0][1]] = True
            q.append((q[0][0] - 1, q[0][1]))
            col+= 1

        q.pop(0)
    print('End of loop')
    if col > maxSize:
        return (-1, -1, -1, -1)
    return (leftAns, rightAns, upperAns, bottomAns)

print(f'{H} x {W}')
for i in range(0, H):
    for j in range(0, W):
        for g in range(1, 11):
            if i == H // 100 * g * 10 and j == 0:
                print(g * 10, f'%')
        leftAns, rightAns, upperAns, bottomAns = (-1, -1, -1, -1)
        if used[i][j] == False:
            leftAns, rightAns, upperAns, bottomAns = bfs(i, j, maxSize)
        if leftAns != -1:
            sample = before[upperAns:bottomAns, leftAns:rightAns]
            before = cv2.rectangle(before, (leftAns, upperAns), (rightAns, bottomAns), (255, 0, 0), 1)


cv2.imwrite('rectangle.png', before)


cv2.waitKey(0)
