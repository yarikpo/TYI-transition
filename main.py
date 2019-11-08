import cv2
import numpy as np

before = cv2.imread('1.png')
after = cv2.imread('2.png')

result_after = after.copy()

###############WORKING WITH IMAGES##############################################
ask = input('Do you want to work with field?[Y/n]: ')
if ask == 'Y' or ask == 'y':
    print('changintg from bgr to hsv...\n')
    before = cv2.cvtColor(before, cv2.COLOR_BGR2HSV)
    after = cv2.cvtColor(after, cv2.COLOR_BGR2HSV)


    before[:, :, 2] = 120
    before[:, :, 1] = 120
    after[:, :, 2] = 120
    after[:, :, 1] = 120

    print('bluring...')
    # beps = 35
    beps = 35
    before = cv2.GaussianBlur(before, (beps, beps), 0)
    after = cv2.GaussianBlur(after, (beps, beps), 0)

    used = []
    for i in range(0, before.shape[0]):
        used.append([])
        for j in range(0, before.shape[1]):
            used[i].append(False)

    def zones(y, x, val, eps, hsv):
        global used
        q = [(y, x)]
        while len(q) > 0:
            x = q[0][1]
            y = q[0][0]
            used[y][x] = True
            # print(f'y: {y} x: {x} used[{y}][{x}]: {used[y][x]} val: {val}')
            hsv[y][x][0] = val
            if x + 1 < hsv.shape[1] and used[y][x + 1] == False and abs(float(hsv[y][x + 1][0]) - val) <= eps:
                # print(abs(float(hsv[y][x + 1][0]) - val))
                q.append((y, x + 1))
                used[y][x + 1] = True
            if x - 1 >= 0 and used[y][x - 1] == False and abs(float(hsv[y][x - 1][0]) - val) <= eps:
                # print(abs(float(hsv[y][x - 1][0]) - val))
                q.append((y, x - 1))
                used[y][x - 1] = True

            if y + 1 < hsv.shape[0] and used[y + 1][x] == False and abs(float(hsv[y + 1][x][0]) - val) <= eps:
                # print(abs(float(hsv[y + 1][x][0]) - val))
                q.append((y + 1, x))
                used[y + 1][x] = True
            if y - 1 >= 0 and used[y - 1][x] == False and abs(float(hsv[y - 1][x][0]) - val) <= eps:
                # print(abs(float(hsv[y - 1][x][0]) - val))
                q.append((y - 1, x))
                used[y - 1][x] = True
            del q[0]
        return hsv


    print('working with 1st photo...\nLoading...')
    for i in range(0, before.shape[0]):
        for j in range(0, before.shape[1]):
            for g in range(1, 11):
                if i == before.shape[0] // 100 * g * 10 and j == 0:
                    print(g * 10, f'%')
            if used[i][j] == False:
                # print(f"y: {i} - x: {j}")
                before = zones(i, j, before[i][j][0], 15.25, before)

    used = []
    for i in range(0, before.shape[0]):
        used.append([])
        for j in range(0, before.shape[1]):
            used[i].append(False)

    print('working with 2nd photo...\nLoading...')
    for i in range(0, before.shape[0]):
        for j in range(0, before.shape[1]):
            for g in range(1, 11):
                if i == before.shape[0] // 100 * g * 10 and j == 0:
                    print(g * 10, f'%')
            if used[i][j] == False:
                # print(f"y: {i} - x: {j}")
                after = zones(i, j, after[i][j][0], 15.25, after)

    print('changing from hsv to bgr...')
    before = cv2.cvtColor(before, cv2.COLOR_HSV2BGR)
    after = cv2.cvtColor(after, cv2.COLOR_HSV2BGR)

# cv2.imwrite('before_blur.png', before)
# cv2.imwrite('after_blur.png', after)
# exit(code=0)
###############WORKING WITH IMAGES##############################################


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
        for g in range(1, 11):
            if i == H // 100 * g * 10 and j == 0:
                print(g * 10, f'%')
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
        for g in range(1, 11):
            if i == H // 100 * g * 10 and j == 0:
                print(g * 10, f'%')
        leftAns, rightAns, upperAns, bottomAns, col = (-1, -1, -1, -1, -1)
        if used[i][j] == False:
            leftAns, rightAns, upperAns, bottomAns, col = bfs(i, j, maxSize)
        if col > 40 and abs(leftAns - rightAns) > 2 and abs(upperAns - bottomAns) > 2:
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
