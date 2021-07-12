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
    block = [[0]*N for index in range(len(c))]
    for j, c_node in enumerate(c):
        i = 0
        while i < N:
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
            x.append(node['x'])
            y.append(node['y'])
        plt.plot(x, y, color='lightsalmon')


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


def to_grid(s, grid_scale):   # 将点的坐标标准化，用方格中心表示，简化计算
    # 输入s为点集，grid_scale为网格宽度
    grid = []
    for i in range(int(xm/grid_scale)):
        for j in range(int(ym/grid_scale)):
            node = {}
            node['x'] = i*grid_scale + grid_scale/2
            node['y'] = j*grid_scale + grid_scale/2
            grid.append(node)
    s_grid = []
    for si in s:
        for gi in grid:
            d = (si['x'] - gi['x'])**2+(si['y'] - gi['y'])**2
            if d < 2*(grid_scale/2)**2:
                si['x'] = gi['x']
                si['y'] = gi['y']
                break
            else:
                continue
        s_grid.append(si)
    sr_grid = []
    for node in s_grid:
        if node not in sr_grid:
            sr_grid.append(node)
    return sr_grid


def get_distance_to_sink(n, cl):   # n为点集  该函数用于补充测出每个点的基本参数 D_sink
    # D_sink 块距离sink的最近距离，每个块中统一
    dsink_min = [10000000 for index in range(cl)]
    dsink_temp = [0 for index in range(cl)]
    path = [[] for index in range(cl)]
    for index in range(cl):
        for node in n:
            if node['block'] == index:
                dsink_temp[index] = (sink['x'] - node['x'])**2 + (sink['y']-node['y'])**2  # 距离sink的距离
                if dsink_temp[index] < dsink_min[index]:
                    dsink_min[index] = dsink_temp[index]
                    path[index] = [node, sink]
        dsink_min[index] = np.sqrt(dsink_min[index])
        for node in n:
            if node['block'] == index:
                node['D_sink'] = dsink_min[index]
    return n, path


def get_distance_to_seg(n, cl):  # n为点集  该函数用于补充测出每个块的基本参数 D_seg
    # D_seg 块间距的最小值，每个块中统一
    n1 = []
    n2 = []
    dseg = [0 for i in range(cl)]
    conn_path = [[] for i in range(cl)]
    for index in range(cl):
        for node in n:
            if node['block'] == index:
                n1.append(node)
            else:
                n2.append(node)
        data = get_min_path(n1, n2)  # 输出信息  block, link, d_min
        n1 = []
        n2 = []
        conn_path[index] = data[1]
        dseg[index] = data[2]
    for index in range(cl):
        for node in n:
            if node['block'] == index:
                node['D_seg'] = dseg[index]
    return n, conn_path


def get_move_cost(s, cl):
    f1 = [0 for i in range(cl)]
    cost = 0
    path = []
    block_to_sink = []  # 存放到sink的块的信息
    node_to_sink = []   # 存放到sink的点的信息
    block_not_to_sink = []  # 存放 不到sink的块的信息
    node_not_to_sink = []   # 存放 不到sink的点的信息
    for index in range(cl):
        for node in s:
            b = node['block']
            if b == index:
                f1[b] = a1*node['D_sink'] - a2*node['D_seg']
                if f1[b] <= 0:
                    c_temp = node['D_sink']
                    path.append(conn_sink[b])
                    if index not in block_to_sink:
                        cost = cost + c_temp
                        block_to_sink.append(index)
                    node_to_sink.append(node)
                else:
                    # 不能简单地跟周围的块连接起来。。。。
                    if index not in block_not_to_sink:
                        block_not_to_sink.append(index)
                    node_not_to_sink.append(node)
                    continue
    for index in block_not_to_sink:
        bi = []
        for nnt in node_not_to_sink:
            b = nnt['block']
            if b == index:
                bi.append(nnt)
        data = get_min_path(bi, node_to_sink)
        path.append(data[1])
        node_to_sink.extend(bi)
        cost = cost + data[2]
    return cost, path


def get_min_path(b1, b2):  # b1, b2当前要判断的点集
    block = []
    path = [[0,0],[0,0]]
    d_min = 10000000
    link = []
    for node in b1:
        x = node['x']
        y = node['y']
        for anode in b2:
            d = (anode['x'] - x) ** 2 + (anode['y'] - y) ** 2
            if d < d_min:
                d_min = d
                block = [node['block'], anode['block']]
                path = [[anode['x'], anode['y']],[x,y]]
                link = [node, anode]
    return block, link, np.sqrt(d_min)


# PARAMETERS ##############################
xm = 1000    # 横坐标长度
ym = 1000   # 纵坐标长度

# 基站定义
sink = {}
sink['x'] = xm/2  # 基站横坐标
sink['y'] = ym-50  # 基站纵坐标

N = 16   # 每个区域的节点个数
R = 50  # 节点通信半径
[w, h] = [25, 25]  # 网格画图长宽，实际宽度为25
D = 2*xm/3   # 近距离块的划分
a1 = 0.2   # 权重a1 ---> 健壮性和负载均衡
a2 = 1-a1    # a2 ----> 连接成本
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

# 建立一个块的字典数组，主要关注块的操作，但同时还有点的操作，可以把点的数据存在块的某个属性中

# 2021 移动小车网络：健壮性、负载均衡性
S_15_grid = to_grid(S_15, w)
B_15, S_15 = get_border(S_15_grid)  # 边界点
B_15_sorted = sort_border(B_15, len(C_15))  # 按区域划分的二维边界点集合
S_15, conn_sink = get_distance_to_sink(S_15, len(C_15))
S_15, conn_segm = get_distance_to_seg(S_15, len(C_15))
cost_2021, conn_path_selected = get_move_cost(S_15, len(C_15))
print(cost_2021)

# 作图  ####################################
gdf = make_mesh([0, 0, xm, ym], w, h)
gdf.boundary.plot()
draw_nodes(S_15)
draw_line(conn_path_selected)
plt.plot(sink['x'], sink['y'], 'rp')  # 绘制sink点
plt.annotate('sink', xy=(sink['x'], sink['y']), xytext=(-20, 10),
             textcoords='offset points', fontsize=12, color='r')

plt.show()
