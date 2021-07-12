import numpy as np
import matplotlib.pyplot as plt
import random
import geopandas as gpd
from operator import itemgetter
from shapely.geometry import Polygon
from scipy.spatial import distance


# 按照w,h对box进行网格划分 #################
def make_mesh(box, w, h):
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


# 将数组元素解析成字典 ###############
def to_obj(arr):
    obj = [{} for i in  range(len(arr))]
    for index, item in enumerate(arr):
        obj[index]['x'] = item[0]
        obj[index]['y'] = item[1]
        obj[index]['block'] = 0   # 所属区域块编号
        obj[index]['type'] = 'N'  # N为普通点，B为边界点
        obj[index]['movable'] = False   # 是否可移动
    return obj


# 迭代生成子点 #################
def init(radius, center):
    x_center = center['x']
    y_center = center['y']
    while True:
        x = x_center + (random.random() - 0.5) * radius * 2
        y = y_center + (random.random() - 0.5) * radius * 2
        if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2 and 0 < x < xm and 0 < y < ym:
            return x, y


# 判断边界点 ###################
def get_border(nodes):
    border_nodes = []  # 用来存放边界点们
    for index, node in enumerate(nodes):
        node['ngb'] = 0
        angle_r_all = []
        right_max = 0
        S2 = nodes[:]
        del S2[index]  # 中间变量，存放除本节点外的其他点
        for other_node in S2:
            if np.abs(other_node['x'] - node['x']) < 2 * R and np.abs(other_node['y'] - node['y']) < 2 * R:
                d = np.sqrt((node['x'] - other_node['x']) ** 2 + (node['y'] - other_node['y']) ** 2)
                if d < 2 * R:
                    angle_d = np.arccos(((d / 2) / R))
                    if node['x'] > other_node['x']:
                        center_angle = np.arctan((other_node['y'] - node['y']) / (other_node['x'] - node['x'])) + np.pi
                    elif node['x'] == other_node['x']:
                        if node['y'] > other_node['y']:
                            center_angle = -np.pi / 2
                        else:
                            center_angle = np.pi / 2
                    else:
                        center_angle = np.arctan((other_node['y'] - node['y']) / (other_node['x'] - node['x']))
                    # 左右边界
                    left = center_angle - angle_d
                    right = center_angle + angle_d
                    angle_r = [left, right]
                    angle_r_all.append(angle_r)
                    if d < R:   # 加入一个邻居判断
                        node['ngb'] = node['ngb'] + 1

        angle_r_all = sorted(angle_r_all, key=itemgetter(0))  # 按left升序排序
        if angle_r_all[0][0] >= -np.pi / 2:
            border_nodes.append(node)
            node['type'] = 'B'
            continue
        for i in range(0, len(angle_r_all)):
            if angle_r_all[i][0] > right_max:  # 当左端值不与右端当前最大值发生重合时，判为边界点
                border_nodes.append(node)
                node['type'] = 'B'
                break
            if angle_r_all[i][1] > right_max:  # 如果此右端点大于右端当前最大值，进行迭代
                right_max = angle_r_all[i][1]
        if right_max < np.pi * 3 / 2:
            border_nodes.append(node)
            node['type'] = 'B'
    return border_nodes, nodes


def create_nodes(c):  # c母点集，返回生成的所有点S， 以及block：按区域划分的二维点集
    s = []
    block = [[0]*n for index in range(len(c))]
    for j, c_node in enumerate(c):
        i = 0
        while i < n:
            i = i+1
            new_node = {}
            new_node['x'],new_node['y'] = init(R, c_node)
            new_node['block'] = j
            new_node['type'] = 'N'  # N为普通点，B为边界点
            new_node['movable'] = False   # 是否可移动
            block[j][i - 1] = c_node
            c_node = new_node
            s.append(new_node)
    return s, block


def draw_nodes(nodes):  # 画所有点
    for i in nodes:
        if i['movable']:
            plt.plot(i['x'], i['y'], 'mv')
        elif i['type'] == 'B' or i['type'] == 'R':
            plt.plot(i['x'], i['y'], 'c.')
        else:
            plt.plot(i['x'], i['y'], 'k.')


def draw_line(paths):
    for index, path in enumerate(paths):
        x = []
        y = []
        for node in path:
            x.append(node[0])
            y.append(node[1])
        plt.plot(x, y, color='c')


def draw_arrow(paths):
    for path in paths:
        x1 = path[0]['x']
        y1 = path[0]['y']
        x2 = path[1]['x']
        y2 = path[1]['y']
        plt.arrow(x1,y1, x2-x1, y2-y1,length_includes_head=True,
                  head_width=0.8, head_length=1,
                  fc='lightsalmon', ec='lightsalmon')


def sort_border(border, cl):
    border2 = [[] for index in range(cl)]  # cl母点集长度
    for node in border:
        bi = node['block']
        border2[bi].append(node)
    return border2


def desired_node_location(node_path):
    desired_node = []
    x = node_path[0]['x']
    y = node_path[0]['y']
    xs = node_path[1]['x']
    ys = node_path[1]['y']
    xd = xs - x
    yd = ys - y
    t = min(np.abs(0.7 * R / xd), np.abs(0.7 * R / yd))
    while True:
        x1 = x + xd * t
        y1 = y + yd * t
        if np.abs(x1 - node_path[0]['x']) < np.abs(x1 - xs):
            # 靠近x1 则属于第一个点所在的block，在寻找中继时，去该block中寻找，为减小计算量
            desired_node.append({'x': x1, 'y': y1, 'block': node_path[0]['block'], 'type': 'D', 'movable': False})
        else:
            desired_node.append({'x': x1, 'y': y1, 'block': node_path[1]['block'], 'type': 'D', 'movable': False})
        x = x1
        y = y1
        if np.abs(x - xs) < 0.7 * R and np.abs(y - ys) < 0.7 * R:
            break
    return desired_node


# 关于2018圆桌协议算法的相关函数 ###################################
def get_d_2018(border):   # blocks 按区域划分的二维点集
    # 获取所有边界点到达圆桌的距离
    d_all = 0
    for nodes in border:
        d_min = 1000000
        for node in nodes:
            d = (node['x']-xm/2)**2 + (node['y']-ym/2)**2
            node['d'] = d
            if d < d_min:
                d_min = d
            else:
                continue
        d_all = d_all + np.sqrt(d_min) - R   # 圆桌协议过程中移动的全路径
    return border, d_all


def get_min_path_2018(border):
    b1 = []
    b2 = border[:]
    conn_path = []
    conn_block = []  #
    conn_node = []  # 相连的两点信息
    conn_id = 0
    while True:
        for node in border:
            if node['block'] == conn_id:
                b1.append(node)
                b2.remove(node)
                if len(b2) == 0:
                    return conn_block, conn_path, conn_node
            else:
                pass
        data = get_min_path(b1, b2)
        if (data[1][0][0] - data[1][1][0])**2 +(data[1][0][1] - data[1][1][1])**2 > R*R:
            conn_path.append(data[1])
            conn_block.append(data[0])
            conn_node.append(data[2])
        conn_id = data[0][1]
    return 0


def get_min_path(b1, b2):  # b1, b2当前要判断的点集
    block= []
    path = [[0,0],[0,0]]
    d_min = 100000
    link = []
    for node in b1:
        x = node['x']
        y = node['y']
        for anode in b2:
            d = (anode['x'] - x) ** 2 + (anode['y'] - y) ** 2
            if d < d_min:
                d_min = d
                block = [node['block'],anode['block']]
                path = [[anode['x'], anode['y']],[x,y]]
                link = [node, anode]
    return block, path, link


def get_relay_2018(s, b, n):   # 识别Relay，即每个区域用于连接其他区域的关键节点
    for i, item in enumerate(n):
        if item[0] in s:
            id = s.index(item[0])
            s[id]['type'] = 'R'
            s[id]['conn'] = b[i][1]
        if item[1] in s:
            id = s.index(item[1])
            s[id]['type'] = 'R'
            s[id]['conn'] = b[i][0]
    return s


def desired_node_location_2018(n):  # 获取连接线路上的路径点
    desired_node = []
    for node_path in n:
        desired_node.extend(desired_node_location(node_path))
    return desired_node


def get_replace_cost(desired_node, ba):   # 给每个路径上选定的位置，匹配一个node
    # ba：区域里的所有点（二维集合）
    badr = []  # ba delete relay
    cost2 = 0
    path = []
    test = []  # 用来测试去除点后是否能保证连通性
    for item in ba:
        # 先把关键中继R从点集中去除
        if item['type'] == 'B' or item['type'] == 'N':
            badr.append(item)
        else:
            pass
    badr.sort(key=lambda bo: bo['ngb'])
    for dn in desired_node:
        able_nodes = []
        bi = dn['block']
        for node in badr:
            if node['block'] == bi:
                d = (node['x'] - dn['x'])**2+(node['y']-dn['y'])**2
                node['cost'] = np.sqrt(d)
                able_nodes.append(node)
        able_nodes.sort(key=lambda ab: ab['cost'])  # 每个desire_node对应block中可行的点
        able_nodes_pro = able_nodes[:]
        for item in able_nodes:
            test = able_nodes[:]
            protect = True
            test.remove(item)
            for i1 in test:
                t2 = test[:]
                if i1 in t2:
                    t2.remove(i1)
                for i2 in t2:
                    d = (i1['x']-i2['x'])**2 + (i1['y']-i2['y'])**2
                    if d < R*R:   # i2连通√
                        protect = True
                        break
                    else:
                        protect = False
                if not protect:  # 遍历完发现i1与块不连通，说明这个测试点需要去除
                    if item in able_nodes_pro:
                        able_nodes_pro.remove(item)
                    break
        cost2 = cost2 + able_nodes_pro[0]['cost']
        path.append([able_nodes_pro[0], dn])
        if able_nodes_pro[0] in badr:
            badr.remove(able_nodes_pro[0])
    return cost2, path


# PARAMETERS ##############################
xm = 1000    # 横坐标长度
ym = 1000   # 纵坐标长度
sink = {'x': 0, 'y': 0}   # 基站定义
sink['x'] = xm/2  # 基站横坐标
sink['y'] = ym-50  # 基站纵坐标
n = 16   # 每个区域的节点个数
R = 50  # 节点通信半径
[w, h] = [50, 50]  # 网格长宽
# END OF PARAMETERS ########################

# 人为指定中心点 ###########################
C_20 = [(50, 100), (100, 400), (50, 700), (30, 950), (300, 30), (340, 350), (260, 680), (280, 890),
        (500, 200), (500, 550), (590, 720), (570, 900), (730, 100), (600, 400), (780, 830),
        (950, 30), (840, 300), (870, 500), (980, 700), (950, 950)]
C_20 = to_obj(C_20)

C_15 = [(50, 100), (100, 400), (50, 700), (50, 900), (340, 340), (400, 660),
        (500, 950), (500, 200), (590, 720), (730, 100), (650, 480), (780, 830), (890, 30), (870, 400), (900, 950)]
C_15 = to_obj(C_15)

# 当区域数为15时 ###########################
S_15, block_15 = create_nodes(C_15)

B_15, S_15 = get_border(S_15)  # 边界点
B_15_sorted = sort_border(B_15, len(C_15))  # 按区域划分的二维边界点集合


# 2018 圆桌协议相关
block_15_2018, d_cost_2018_1 = get_d_2018(B_15_sorted)
conn_block_2018, conn_path_2018, conn_node_2018= get_min_path_2018(B_15)
S_15_2018 = get_relay_2018(S_15, conn_block_2018, conn_node_2018)
DN = desired_node_location_2018(conn_node_2018)
d_cost_2018_2, move_path_2018 = get_replace_cost(DN, S_15)
cost_2018 = d_cost_2018_1 + d_cost_2018_2
print(d_cost_2018_1, d_cost_2018_2, cost_2018)

# 作图  ####################################
gdf = make_mesh([0, 0, xm, ym], w, h)
gdf.boundary.plot()
draw_nodes(S_15)
draw_line(conn_path_2018)
draw_nodes(DN)
draw_arrow(move_path_2018)
plt.plot(sink['x'], sink['y'], 'rp')  # 绘制sink点
plt.annotate('sink', xy=(sink['x'], sink['y']), xytext=(-20, 10),
             textcoords='offset points', fontsize=12, color='r')

plt.show()
