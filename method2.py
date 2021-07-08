import numpy as np
import matplotlib.pyplot as plt
import random
import geopandas as gpd
from operator import itemgetter
from shapely.geometry import Polygon


def make_mesh(box, w, h):  # 按照w,h对box进行网格划分
    [xmin, ymin, xmax, ymax] = box
    list_x=np.arange(xmin, xmax, w)
    list_y=np.arange(ymin, ymax, h)
    polygon_list = []
    for i in range(len(list_x)):
        for j in range(len(list_y)):
            xleft = list_x[i]
            ydown = list_y[j]
            if i == len(list_x)-1:
                xright = xmax
            else:
                xright = list_x[i+1]
            if j == len(list_y)-1:
                yup = ymax
            else:
                yup = list_y[j+1]
            rectangle = Polygon([(xleft, ydown), (xright, ydown), (xright, yup), (xleft, yup)])
            polygon_list.append(rectangle)

    return gpd.GeoSeries(polygon_list)


def init(radius, center):
    x_center = center[0]
    y_center = center[1]
    while True:
        x = x_center + (random.random() - 0.5) * radius * 2
        y = y_center + (random.random() - 0.5) * radius * 2
        if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2 and 0 < x < xm and 0 < y < ym:
            return [x, y]


# 判断边界点 ###################
def get_border(nodes):
    border_nodes = []  # 用来存放边界点们
    for index, node in enumerate(nodes):
        angle_r_all = []
        right_max = 0
        S2 = S[:]
        del S2[index]  # 中间变量，存放除本节点外的其他点
        for other_node in S2:
            if np.abs(other_node[0] - node[0]) < 2 * R and np.abs(other_node[1] - node[1]) < 2 * R:
                d = np.sqrt((node[0] - other_node[0]) ** 2 + (node[1] - other_node[1]) ** 2)
                if d < 2 * R:
                    angle_d = np.arccos(((d / 2) / R))
                    if node[0] > other_node[0]:
                        center_angle = np.arctan((other_node[1] - node[1]) / (other_node[0] - node[0])) + np.pi
                    elif node[0] == other_node[0]:
                        if node[1] > other_node[1]:
                            center_angle = -np.pi / 2
                        else:
                            center_angle = np.pi / 2
                    else:
                        center_angle = np.arctan((other_node[1] - node[1]) / (other_node[0] - node[0]))
                    # 左右边界
                    left = center_angle - angle_d
                    right = center_angle + angle_d
                    angle_r = [left, right]
                    angle_r_all.append(angle_r)

        angle_r_all = sorted(angle_r_all, key=itemgetter(0))  # 按left升序排序
        if angle_r_all[0][0] >= -np.pi / 2:
            border_nodes.append(node)
            continue
        for i in range(0, len(angle_r_all)):
            if angle_r_all[i][0] > right_max:  # 当左端值不与右端当前最大值发生重合时，判为边界点
                border_nodes.append(node)
                break
            if angle_r_all[i][1] > right_max:  # 如果此右端点大于右端当前最大值，进行迭代
                right_max = angle_r_all[i][1]
        if right_max < np.pi * 3 / 2:
            border_nodes.append(node)
    return border_nodes


# PARAMETERS ##############################
xm = 1000    # 横坐标长度
ym = 1000   # 纵坐标长度
sink = {'x': 0, 'y': 0}   # 基站定义
sink['x'] = xm/2  # 基站横坐标
sink['y'] = ym-10  # 基站纵坐标
n = 16   # 每个区域的节点个数
R = 30  # 节点通信半径
[w, h] = [50, 50]  # 网格长宽
# END OF PARAMETERS ########################

# 人为指定中心点 ###########################
C_20 = [(50, 100), (100, 400), (50, 700), (30, 950), (300, 30), (340, 350), (260, 680), (280, 890),
        (500, 200), (500, 550), (590, 720), (570, 900), (730, 100), (600, 400), (780, 830),
        (950, 30), (840, 300), (870, 500), (980, 700), (950, 950)]

C_15 = [(50, 100), (100, 400), (50, 700), (50, 900), (340, 340), (400, 660),
        (500, 950), (500, 200), (590, 720), (730, 100), (650, 480), (780, 830), (890, 30), (870, 400), (900, 950)]

S = C_15[:]  # 节点集
for c_node in C_15:
    i = 0
    while i < n:
        i = i+1
        new_node = init(R, c_node)
        c_node = new_node
        S.append(new_node)

B = get_border(S)  # 边界点

# 作图  ####################################
gdf = make_mesh([0, 0, xm, ym], w, h)
gdf.boundary.plot()
for i in S:
    plt.plot(i[0], i[1], 'k.')

for i in B:
    plt.plot(i[0], i[1], 'b.')

plt.plot(sink['x'], sink['y'], 'rp')  # 绘制sink点
plt.annotate('sink', xy=(sink['x'], sink['y']), xytext=(-20, 10),
             textcoords='offset points', fontsize=12, color='r')

plt.show()
