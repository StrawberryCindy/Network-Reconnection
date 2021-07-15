import numpy as np
import matplotlib.pyplot as plt
import random
import geopandas as gpd
from operator import itemgetter
from shapely.geometry import Polygon


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
def init(radius, center, xm, ym):
    x_center = center['x']
    y_center = center['y']
    while True:
        x = x_center + (random.random() - 0.5) * radius * 2
        y = y_center + (random.random() - 0.5) * radius * 2
        if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2 and 0 < x < xm and 0 < y < ym:
            return x, y


# 判断边界点 ###################
def get_border(nodes, R):
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


def create_nodes(c, n, R, xm, ym):  # c母点集，返回生成的所有点S， 以及block：按区域划分的二维点集
    s = []
    block = [[0]*n for index in range(len(c))]
    for j, c_node in enumerate(c):
        i = 0
        while i < n:
            i = i+1
            new_node = {}
            new_node['x'],new_node['y'] = init(R, c_node, xm, ym)
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
            x.append(node['x'])
            y.append(node['y'])
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


def get_move_2016(s, blocks, mob_n, n):  # blocks 按区域划分的二维点集，mob_n区域中的可移动节点数
    for i in range(0, len(blocks)):
        j = 0
        cant_find = 0
        while j < mob_n:
            index = int(random.random()*n)
            if s[index+i*n]['type'] != 'B':
                s[index + i*n]['movable'] = True
                j = j+1
            else:
                if cant_find == 10:
                    break
                else:
                    cant_find = cant_find+1
                    continue
    return s


def sort_border(border, cl):
    border2 = [[] for index in range(cl)]  # cl母点集长度
    for node in border:
        bi = node['block']
        border2[bi].append(node)
    return border2


def get_min_path_2016(border, m, R):  # 获取的cost最小的路径
    b1 = []
    b2 = border[:]
    conn_path = []
    conn_id = 0
    cost = 0
    move_path = []
    dn = []
    r = []   # 每找到一条路径后取出的可移动点
    while True:
        for node in border:
            if node['block'] == conn_id:
                b1.append(node)
                if node in b2:
                    b2.remove(node)
                if len(b2) == 0:
                    return conn_path, cost, move_path, dn
            else:
                pass
        m = [i for i in m if i not in r]
        data = get_path_blcok(b1, b2, m, R)  # 获取的数据格式conn_path, cost_all_min, move_path, desired_node, r
        if data[2] == 0:
            return 0,0,0,0
        else:
            conn_path.append(data[0])
            cost = cost + data[1]
            move_path.append(data[2])
            conn_id = data[0][1]['block']
            dn.extend(data[3])
            r.extend(data[4])


def get_path_blcok(b1, b2, m, R):
    # 输入：边界点集b1, b2, m可移动点，
    # 输出：conn_path，cost（最小值，唯一哦）, move_path:[]数组（因为有多个desired_node)
    conn_path = {}
    cost_all_min = 100000
    move_path_all = []
    desired_node = []
    r = []
    for n1 in b1:
        for n2 in b2:
            dn = desired_node_location([n1, n2], R)
            move_path, cost_all, r_test = get_path_node(dn, m)
            if move_path == 0:
                continue
            else:
                if move_path in move_path_all:
                    continue
                else:
                    move_path_all.append({'path': move_path, 'cost': cost_all})
                    if cost_all < cost_all_min:
                        desired_node = dn
                        cost_all_min = cost_all
                        conn_path = [n1, n2]
                        r = r_test

    for item in move_path_all:
        if item['cost'] == cost_all_min:
            move_path = item['path']

    return conn_path, cost_all_min, move_path, desired_node, r


def get_path_node(dn, m):
    mdr = m[:]
    cost_all = 0
    move_path = []
    r = []
    for dni in dn:
        cost_min = 10000
        move_path_test = []
        r_test = {}
        for mk in mdr:
            cost = np.sqrt((mk['x'] - dni['x']) ** 2 + (mk['y'] - dni['y']) ** 2)
            if cost < cost_min:
                cost_min = cost
                move_path_test = [mk, dni]
                if mk in mdr:
                    mdr.remove(mk)
                    r_test = mk
        if len(move_path_test) == 0:  # 可移动点都不够了 直接舍去
            return 0,0,0
        else:
            move_path.append(move_path_test)
            r.append(r_test)
            cost_all = cost_all + cost_min
    return move_path, cost_all, r


def desired_node_location(node_path, R):
    desired_node = []
    x = node_path[0]['x']
    y = node_path[0]['y']
    xs = node_path[1]['x']
    ys = node_path[1]['y']
    xd = xs - x
    yd = ys - y
    if xd == 0 or yd == 0:
        if xd == 0 and yd == 0:
            return []
        elif yd == 0:
            t = np.abs(0.7 * R / xd)
        elif xd == 0:
            t = np.abs(0.7 * R / yd)
    else:
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


# PARAMETERS ##############################
xm = 1000    # 横坐标长度
ym = 1000   # 纵坐标长度
sink = {'x': 0, 'y': 0}   # 基站定义
sink['x'] = xm/2  # 基站横坐标
sink['y'] = ym-50  # 基站纵坐标
N = 16   # 每个区域的节点个数
R = 50  # 节点通信半径
[w, h] = [50, 50]  # 网格长宽

Dp = 0.1      # 随机破坏节点比例
# END OF PARAMETERS ########################

'''
# 人为指定中心点 ###########################
C_20 = [(50, 100), (100, 400), (50, 700), (30, 950), (300, 30), (340, 350), (260, 680), (280, 890),
        (500, 200), (500, 550), (590, 720), (570, 900), (730, 100), (600, 400), (780, 830),
        (950, 30), (840, 300), (870, 500), (980, 700), (950, 950)]
C_20 = to_obj(C_20)

C_15 = [(50, 100), (100, 400), (50, 700), (50, 900), (340, 340), (400, 660),
        (500, 950), (500, 200), (590, 720), (730, 100), (650, 480), (780, 830), (890, 30), (870, 400), (900, 950)]
C_15 = to_obj(C_15)


# 当区域数为15时 ###########################
S_15, block_15 = create_nodes(C_15, N, R, xm, ym)

B_15, S_15 = get_border(S_15, R)  # 边界点
B_15_sorted = sort_border(B_15, len(C_15))  # 按区域划分的二维边界点集合

# 2016 移动中继相关
S_15_2016 = get_move_2016(S_15, block_15, 9, N)
M_15 = []
for node in S_15_2016:
    if node['movable']:
        M_15.append(node)
conn_path_2016, cost_2016, move_path_2016, DN = get_min_path_2016(B_15, M_15, R)
if cost_2016 == 0:
    print('没有找到可行的路径')
else:
    pass
    print(cost_2016)
    # S1 = S.extend(DN)
    # S2 = destroy(S1, Dp)
    # get_node_conn(S2)
    # 作图  ####################################
    gdf = make_mesh([0, 0, xm, ym], w, h)
    gdf.boundary.plot()
    draw_nodes(S_15)
    draw_line(conn_path_2016)
    draw_nodes(DN)
    for mp in move_path_2016:
        draw_arrow(mp)
    plt.plot(sink['x'], sink['y'], 'rp')  # 绘制sink点
    plt.annotate('sink', xy=(sink['x'], sink['y']), xytext=(-20, 10),
                 textcoords='offset points', fontsize=12, color='r')

    plt.show()
'''
