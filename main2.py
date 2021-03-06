# -*- coding: utf-8 -*-

import math
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import matplotlib.patches as patches
import numpy as np


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def GetIntersectPoint(AP1, AP2, BP1, BP2):
    under = (BP2.y - BP1.y) * (AP2.x - AP1.x) - (BP2.x - BP1.x) * (AP2.y - AP1.y)
    if under == 0:
        return False
    _t = (BP2.x - BP1.x) * (AP1.y - BP1.y) - (BP2.y - BP1.y) * (AP1.x - BP1.x)
    _s = (AP2.x - AP1.x) * (AP1.y - BP1.y) - (AP2.y - AP1.y) * (AP1.x - BP1.x)
    t = _t / under
    s = _s / under
    if t < 0. or t > 1. or s < 0. or s > 1.:
        return False
    if _t == 0 and _s == 0:
        return False
    return True


def onSegment(p, q, r):
    if ((q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
            (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
        return True
    return False


def orientation(p, q, r):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Colinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise

    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
    # for details of below formula.

    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
    if (val > 0):

        # Clockwise orientation
        return 1
    elif (val < 0):

        # Counterclockwise orientation
        return 2
    else:

        # Colinear orientation
        return 0


# The main function that returns true if
# the line segment 'p1q1' and 'p2q2' intersect.
def doIntersect(p1, q1, p2, q2):
    # Find the 4 orientations required for
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if ((o1 != o2) and (o3 != o4)):
        return True

    # Special Cases

    # p1 , q1 and p2 are colinear and p2 lies on segment p1q1
    if ((o1 == 0) and onSegment(p1, p2, q1)):
        return True

    # p1 , q1 and q2 are colinear and q2 lies on segment p1q1
    if ((o2 == 0) and onSegment(p1, q2, q1)):
        return True

    # p2 , q2 and p1 are colinear and p1 lies on segment p2q2
    if ((o3 == 0) and onSegment(p2, p1, q2)):
        return True

    # p2 , q2 and q1 are colinear and q1 lies on segment p2q2
    if ((o4 == 0) and onSegment(p2, q1, q2)):
        return True

    # If none of the cases
    return False


def is_divide_pt(x11, y11, x12, y12, x21, y21, x22, y22):
    f1 = (x12 - x11) * (y21 - y11) - (y12 - y11) * (x21 - x11)
    f2 = (x12 - x11) * (y22 - y11) - (y12 - y11) * (x22 - x11)
    if f1 * f2 < 0:
        return True
    else:
        return False


def is_cross_pt(x11, y11, x12, y12, x21, y21, x22, y22):
    b1 = is_divide_pt(x11, y11, x12, y12, x21, y21, x22, y22)
    b2 = is_divide_pt(x21, y21, x22, y22, x11, y11, x12, y12)
    if b1 and b2:
        return True


startx, starty = map(int, raw_input('?????? ????????? ???????????????: ').split())
a = float(input('???????????? ????????? ???????????????: '))
node_num = int(input('searchnode??? ?????? ???????????????: '))
# searchnode??? ????????? ??????
points = []

i = 0
while i < node_num:
    tempx, tempy = map(int, raw_input('node ????????? ???????????????').split())
    points.append([tempx, tempy])
    i += 1
# search node??? x,y ?????? ?????? ????????? ????????????
leng = len(points)
maxx = 0
maxy = 0
minx = 999
miny = 999
for point in points:
    if maxx < point[0]:
        maxx = point[0]
    if maxy < point[1]:
        maxy = point[1]
    if minx > point[0]:
        minx = point[0]
    if miny > point[1]:
        miny = point[1]
# coord: ????????? ????????? ???????????? ??????

coord = []
beginx = 0
if startx > minx:
    beginx = minx - startx - 1
else:
    beginx = int(-startx / a) - 1
endx = int(maxx / a) + 1
beginy = 0
endy = int(2 * maxy / a) + 1
if starty > miny:
    beginy = miny - starty - 1
else:
    beginy = int(-starty / a) - 1

count = 0
label = []
width = endx - beginx
height = endy - beginy
array = []
i = beginy
while i < endy:
    j = beginx
    while j < endx:
        coord.append([j * a + a / 2, a * i - a / 2, 0])
        j += 1
        label.append(count)
        count += 1
    i += 1
i = beginy
while i < endy:
    j = beginx
    while j < endx:
        coord.append([a * j, a * i, 0])
        label.append(count)
        count += 1
        j += 1
    i += 1
count = 0
p = width * height + 1
m = width * (height - 1)
while count < len(label):
    if count == width - 1:  # ?????? ?????? ?????????
        array.append([-1, count + width, count + p - 1, -1, -1, -1])
    elif count == width * (2 * height - 1):  # ?????? ?????? ?????????
        array.append([-1, -1, -1, -1, count - width, count - m - width])
    elif count == width * height:  # ?????? ?????? ?????????
        array.append([count - m, count + width, -1, -1, -1, count + m - width])
    elif count == width * height - 1:  # ?????? ?????? ?????????
        array.append([-1, -1, count + p - 1, count + p - 1 - width, count - width, -1])
    elif count / width == 0:  # ?????? ?????????1
        array.append([count + p, count + width, count + p - 1, -1, -1, -1])
    elif count / width == height:  # ?????? ?????????2
        array.append(
            [count - m, count + width, count - m - 1, count - m - 1 - width, -1, count - m - width])
    elif count / width == 2 * height - 1:  # ????????????
        array.append([-1, -1, -1, count - m - 1 - width, count - width, count - m - width])
    elif count / width == height - 1:  # ????????????2
        array.append([count + p, -1, count + p - 1, count + p - 1 - width, count - width, count + p - width])
    elif count % width == 0 and count > width * height:  # ?????? ?????????
        array.append([count - m, count + width, -1, -1, count - width, count - m - width])
    elif count % width == (width - 1) and count < width * height:  # ????????? ?????????
        array.append([-1, count + width, count + p - 1, count + p - 1 - width, count - width, -1])
    elif count > width * height:
        array.append([count - m, count + width, count - m - 1, count - m - 1 - width, count - width, count - m - width])
    elif count < width * height:
        array.append([count + p, count + width, count + p - 1, count + p - 1 - width, count - width, count + p - width])
    count += 1
# ?????? ????????? ?????? ???
fig, ax = plt.subplots(1)
ax.set_aspect('equal')

# coord??? ???????????? ????????? ????????? x????????? ?????????.
hcoord = []
for c in coord:
    hcoord.append(c[0])
# coord??? ???????????? ????????? ????????? y????????? ?????????.
vcoord = [2. * np.sin(np.radians(60)) * (c[1] - c[2]) / 3.
          for c in coord]

# searchnode??? ????????? ???????????? ?????????.
check_poly = patches.Polygon(points, edgecolor='black', fill=False)
# searchnode??? ?????? ???????????? ?????????.
ax.add_patch(check_poly)

# greenhex:searchnode???????????? ????????? ????????????????????? ????????? ??????, redhex:????????? ????????????????????? ????????? ??????
greenhex = []
redhex = []
N_adj_i = {}
N = {}
# ?????? ??????????????? ????????? redhex??? greenhex??? ????????????.
for x, y, la in zip(hcoord, vcoord, label):
    check = False
    x += startx
    y += starty

    if x > 0 and y > 0:
        for index in range(6):
            ta = array[la][index]
            if ta < 0:
                continue
            elif hcoord[ta] + startx > 0 and vcoord[ta] + starty > 0:
                continue
            else:
                array[la][index] = -1
        hex = RegularPolygon((x, y), numVertices=6, radius=2. / 3. * a / 2, orientation=np.radians(30), alpha=0.2,
                             edgecolor='k', facecolor='red')
        # check_points??? ???????????? ?????? ???????????? ?????? greenhex
        tpoints = []
        t = 0
        while (t < 6):
            tx = x + hex.radius * math.cos(math.pi / 3 * t)
            ty = y + hex.radius * math.sin(math.pi / 3 * t)
            tpoints.append([tx, ty])
            t += 1
        for p in tpoints:
            checknum = 0
            k = 0;
            while k < leng:
                p1 = Point(0, p[1])
                q1 = Point(p[0], p[1])
                p2 = Point(points[k][0], points[k][1])
                q2 = Point(points[(k + 1) % leng][0], points[(k + 1) % leng][1])
                if GetIntersectPoint(p1, q1, p2, q2):
                    checknum += 1
                k += 1
            if checknum % 2 == 1:
                hex.set_color('green')
                check = True
                greenhex.append(la)
                break

        if check is False:
            k = 0
            while k < 6:
                l = 0
                while l < leng:
                    p1 = Point(tpoints[k][0], tpoints[k][1])
                    q1 = Point(tpoints[(k + 1) % 6][0], tpoints[(k + 1) % 6][1])
                    p2 = Point(points[l][0], points[l][1])
                    q2 = Point(points[(l + 1) % leng][0], points[(l + 1) % leng][1])
                    if GetIntersectPoint(p1, q1, p2, q2):
                        hex.set_color('green')
                        check = True
                        greenhex.append(la)
                        break
                    l += 1
                if check is True:
                    break
                k += 1
        # ????????? ?????? redhex
        if check is False:
            redhex.append(la)
        ax.add_patch(hex)
        ax.text(x, y + 0.2, la, ha='center', va='center', size=10 * a)
        N[la] = [x, y]
        N_adj_i[la] = array[la]
# ????????? ????????? ????????????.
startdot = patches.Circle((startx, starty), 0.3)
ax.add_patch(startdot)
ax.autoscale()
# selectnodes??? ?????????????????? ????????? ????????? ????????? ????????? map??? ????????? ????????????.
if (maxx < startx):
    maxx = startx
if (maxy < starty):
    maxy = starty
# greenhex??? redhex??? ????????? ????????? ????????????.
print(N_adj_i)
print(N)
print("greenhex")
print(greenhex)
print("redhex")
print(redhex)

plt.axis([0, maxx + startx + a / 2, 0, maxy + starty + a / 2])
plt.show()
