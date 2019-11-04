import cv2
import numpy as np

before = cv2.imread('1.png')
after = cv2.imread('2.png')

result_after = after.copy()

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

def check2(y1, x1, y2, x2):
    global before, after
    if after[y1][x1][0] != after[y2][x2][0] or after[y1][x1][1] != after[y2][x2][1] or after[y1][x1][2] != after[y2][x2][2]:
        return False
    return True

def bfs(y, x, maxSize):
    # print('bfs!')
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
        # if col % 10000 == 0:
        #     print(col)
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
    # print('End of loop')
    if col > maxSize:
        return (-1, -1, -1, -1, -1)
    return (leftAns, rightAns, upperAns, bottomAns, col)

###############################################################################


def bfs2(y, x, maxSize):
    # print('bfs!')
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
        # if col % 10000 == 0:
        #     print(col)
        leftAns = min(q[0][1], leftAns)
        rightAns = max(q[0][1], rightAns)
        upperAns = min(q[0][0], upperAns)
        bottomAns = max(q[0][0], bottomAns)

        if q[0][1] + 1 < W and check2(q[0][0], q[0][1], q[0][0], q[0][1] + 1) == True and used[q[0][0]][q[0][1] + 1] == False:
            used[q[0][0]][q[0][1] + 1] = True
            q.append((q[0][0], q[0][1] + 1))
            col+= 1
        if q[0][1] - 1 >= 0 and check2(q[0][0], q[0][1], q[0][0], q[0][1] - 1) == True and used[q[0][0]][q[0][1] - 1] == False:
            used[q[0][0]][q[0][1] - 1] = True
            q.append((q[0][0], q[0][1] - 1))
            col+= 1
        if q[0][0] + 1 < H and check2(q[0][0], q[0][1], q[0][0] + 1, q[0][1]) == True and used[q[0][0] + 1][q[0][1]] == False:
            used[q[0][0] + 1][q[0][1]] = True
            q.append((q[0][0] + 1, q[0][1]))
            col+= 1
        if q[0][0] - 1 >= 0 and check2(q[0][0], q[0][1], q[0][0] - 1, q[0][1]) == True and used[q[0][0] - 1][q[0][1]] == False:
            used[q[0][0] - 1][q[0][1]] = True
            q.append((q[0][0] - 1, q[0][1]))
            col+= 1

        q.pop(0)
    # print('End of loop')
    if col > maxSize:
        return (-1, -1, -1, -1, -1)
    return (leftAns, rightAns, upperAns, bottomAns, col)


###############################################################################



print(f'{H} x {W}')

print('loading...')
for i in range(0, H):
    for j in range(0, W):
        # for g in range(1, 11):
        #     if i == H // 100 * g * 10 and j == 0:
        #         print(g * 10, f'%')
        leftAns, rightAns, upperAns, bottomAns, col = (-1, -1, -1, -1, -1)
        if used[i][j] == False:
            leftAns, rightAns, upperAns, bottomAns, col = bfs2(i, j, maxSize)
        if col > 9 and abs(leftAns - rightAns) > 2 and abs(upperAns - bottomAns) > 2:
            sample = before[upperAns:bottomAns, leftAns:rightAns]
            # if i > 2:
            #     cv2.imwrite(f'{i}.png', sample)

            cv2.rectangle(result_after, (leftAns, upperAns), (rightAns, bottomAns), (0, 255, 255), 2)



used = []
maxSize = H * W - 1e5

for i in range(0, H):
    used.append([])
    for j in range(0, W):
        used[i].append(False)

print('loading...')
for i in range(0, H):
    for j in range(0, W):
        # for g in range(1, 11):
        #     if i == H // 100 * g * 10 and j == 0:
        #         print(g * 10, f'%')
        leftAns, rightAns, upperAns, bottomAns, col = (-1, -1, -1, -1, -1)
        if used[i][j] == False:
            leftAns, rightAns, upperAns, bottomAns, col = bfs(i, j, maxSize)
        if col > 9 and abs(leftAns - rightAns) > 2 and abs(upperAns - bottomAns) > 2:
            sample = before[upperAns:bottomAns, leftAns:rightAns]
            # if i > 2:
            #     cv2.imwrite(f'{i}.png', sample)

            # cv2.rectangle(result_after, (leftAns, upperAns), (rightAns, bottomAns), (255, 0, 0), 1)
            # before = cv2.rectangle(before, (leftAns, upperAns), (rightAns, bottomAns), (255, 0, 0), 1)

            img_grey = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
            template = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)

            w, h = template.shape[::-1]
            res = cv2.matchTemplate(img_grey, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.99
            loc = np.where(res >= threshold)
            # print(f'len:', loc[::-1])

            for pt in zip(*loc[::-1]):
                cv2.rectangle(result_after, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
                cv2.line(result_after, (abs(leftAns + rightAns) // 2, abs(upperAns + bottomAns) // 2), (pt[0] + w // 2, pt[1] + h // 2), (0, 0, 0))

            cv2.rectangle(result_after, (leftAns, upperAns), (rightAns, bottomAns), (255, 0, 0), 2)
            if len(loc[0]) == 0:
                cv2.rectangle(result_after, (leftAns, upperAns), (rightAns, bottomAns), (0, 0, 255), 2)



cv2.imwrite('rectangle.png', result_after)
# cv2.imshow('after', after)
# cv2.imshow('before', before)

cv2.waitKey(0)
