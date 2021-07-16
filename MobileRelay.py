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


def create_nodes(c,n,R, xm, ym):  # c母点集，返回生成的所有点S， 以及block：按区域划分的二维点集
    s = []
    block1 = [[0]*n for index in range(len(c))]
    block2 = [{} for index in range(len(c))]
    for j, c_node in enumerate(c):
        i = 0
        c_node['block'] = j
        block2[j]['nodes'] = []
        block2[j]['nodes'].append(c_node)
        s.append(c_node)
        while i < n - 1:
            i = i+1
            new_node = {}
            new_node['x'], new_node['y'] = init(R, c_node,xm, ym)
            new_node['block'] = j
            new_node['type'] = 'N'  # N为普通点，B为边界点
            new_node['movable'] = False   # 是否可移动
            block1[j][i - 1] = c_node
            block2[j]['nodes'].append(new_node)
            block2[j]['id'] = j
            s.append(new_node)
            c_node = new_node
    return s, block1, block2


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


def to_grid(s, grid_scale,xm, ym):   # 将点的坐标标准化，用方格中心表示，简化计算
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


def get_plumpness(blocks, xm,ym, w, h):   # 计算每个区域的参数
    for block in blocks:
        left = xm
        right = 0
        bottom = ym
        top = 0
        nodes = block['nodes']
        for node in nodes:
            if node['x'] < left:
                left = node['x']
            if node['x'] > right:
                right = node['x']
            if node['y'] < bottom:
                bottom = node['y']
            if node['y'] > top:
                top = node['y']
        xd = right - left + w
        yd = top - bottom + h
        l_max = max(xd, yd)
        ideal_grid = (l_max/w)**2
        block['plumpness'] = len(block['nodes']) / ideal_grid
    return blocks


def get_distance_to_sink(n, cl, blocks, sink):   # n为点集  该函数用于补充测出每个点的基本参数 D_sink
    # D_sink 块距离sink的最近距离，每个块中统一
    dsink_min = [0 for index in range(cl)]
    dsink_temp = [0 for index in range(cl)]
    path = [[] for index in range(cl)]
    for index in range(cl):
        for i, node in enumerate(n):
            if node['block'] == index:
                dsink_temp[index] = (sink['x'] - node['x'])**2 + (sink['y']-node['y'])**2  # 距离sink的距离
                if dsink_min[index] == 0:
                    dsink_min[index] = dsink_temp[index]
                    path[index] = [node, sink]
                else:
                    if dsink_temp[index] < dsink_min[index]:
                        dsink_min[index] = dsink_temp[index]
                        path[index] = [node, sink]
        dsink_min[index] = np.sqrt(dsink_min[index])
        for node in n:
            if node['block'] == index:
                node['D_sink'] = dsink_min[index]
                blocks[index]['D_sink'] = dsink_min[index]
    return n, path, blocks


def get_distance_to_segm(n, cl, blocks):  # n为点集  该函数用于补充测出每个块的基本参数 D_seg
    # D_seg 块间距的最小值，每个块中统一
    n1 = []
    n2 = []
    dseg = [0]*cl
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
                blocks[index]['D_seg'] = dseg[index]
    return n, conn_path, blocks


'''
def get_distance_to_segm(aim_block, other_blocks):
    # 判断目标区域与其他已有归属区域的最近距离
    # 一定注意 blocks 和 other_blocks 的赋值
    # 返回连接的最短路径
    path = []
    for block in other_blocks:
        data = get_min_path(aim_block['nodes'], block['nodes'])  # 输出信息  block, link, d_min
        path = data[1]
    return path'''


def step1_2021(blocks, D, a1, a2, R, sink):
    path = []
    path_set = 0
    cost = 0
    for block in blocks:
        block['path_set'] = -1
    for block in blocks:
        d_min = 0
        path_temp = []
        if block['D_sink'] <= D:
            # 为近距离组
            f1 = a1*block['D_sink'] - a2*block['D_seg']
            if f1 <= 0:
                # 为直接连sink的区域
                path_set = path_set + 1   # 确定的要连sink啦，它归属的路径组索引为path_set
                # 表明这个区域已经有归属了，其他都为 -1
                for node in block['nodes']:
                    # 找最近的连接路径 path
                    d_temp = (sink['x'] - node['x'])**2 + (sink['y']-node['y'])**2
                    if d_min == 0 and len(path_temp) == 0:
                        d_min = d_temp
                        path_temp = [node, sink]
                    else:
                        if d_temp<d_min:
                            d_min = d_temp
                            path_temp = [node, sink]
                # print('近距离组到sink的path：',block['nodes'][0], path_temp, np.sqrt(d_min))
                block['path_set'] = path_set
                if d_min > R**2:
                    cost = cost + np.sqrt(d_min)  # d_min其实就是连通代价
                    path.append(path_temp)
        else:
            # 远距离组等待step2
            continue
    # 近距离组中要先遍历出全部连接sink的，才能进行下一步连接（有先后顺序）
    for block in blocks:
        d_min = 0
        path_temp = []
        if block['D_sink'] <= D:
            # 为近距离组
            f1 = a1*block['D_sink'] - a2*block['D_seg']
            if f1 > 0:
                # 非连接sink的近距离组，要连接附近的其他已连接的组
                for other_block in blocks:
                    if other_block['path_set'] != -1 and other_block != block:
                        # 判断其他已连接的组，选最近距离
                        data = get_min_path(block['nodes'], other_block['nodes'])
                        d_temp = data[2]
                        if d_min == 0 and len(path_temp) == 0:
                            d_min = d_temp
                            path_temp = data[1]
                            # 更新归属信息
                            block['path_set'] = other_block['path_set']
                        else:
                            if d_temp < d_min:
                                d_min = d_temp
                                path_temp = data[1]
                                # 更新归属信息
                                block['path_set'] = other_block['path_set']
                    else:
                        continue
                # print('近距离组到附近区域的path：', block['nodes'][0], other_block['nodes'][0], path_temp, d_min)
                if len(path_temp) != 0 and d_min > R:
                    path.append(path_temp)
                    cost = cost + d_min  # d_min其实就是连通代价
        else:
            # 远距离组等待step2
            continue
    # print('step1 cost:', cost)
    return path, cost


def step2_2021(blocks, D, Dm, Beita1, Beita2):
    # 遍历已连接sink的区域，选择f2最大的区域连接
    # 返回第二步连接的path和cost
    blocks.sort(key=lambda b: b['D_sink'])
    path = []
    cost = 0
    # 先按照到sink的距离进行排序，再检测
    for block in blocks:
        if block['D_sink'] > D:
            # 对远距离组进行操作
            f2_max = 0
            path_temp = []
            cost_temp = 0
            for other_block in blocks:
                if other_block['path_set'] != -1 and other_block != block:
                    # 对于每个远距离组，找最近的有归属path的组
                    data = get_min_path(block['nodes'], other_block['nodes'])
                    f2 = 1 + Beita1 * other_block['plumpness'] - Beita2 * data[2] / Dm
                    if f2 > f2_max:
                        f2_max = f2
                        path_temp = data[1]
                        cost_temp = data[2]
                        block['path_set'] = other_block['path_set']
            # print('远距离组连接：', path_temp, block['path_set'])
            path.append(path_temp)
            cost = cost + cost_temp
    return path, cost


def step3_2021(blocks):
    path = []
    cost = 0
    path_num = [] # 路径组的id集合
    # 先找到有几条路径组
    for block in blocks:
        if block['path_set'] not in path_num:
            path_num.append(block['path_set'])
    # 再往创建好的path_set里面填block
    path_set = [{} for i in range(len(path_num))]
    for index in range(len(path_set)):
        path_set[index] = {'id': index+1, 'load': 0, 'blocks': []}
        for block in blocks:
            if block['path_set'] == index+1:
                path_set[index]['blocks'].append(block)
                path_set[index]['load'] = path_set[index]['load'] + 1
    path_set.sort(key=lambda ps: ps['load'], reverse=True)
    # for item in path_set:
    #    print('path_set:', item['id'], 'load:', item['load'])
    # 选负载最重的一条路
    path_temp = []
    d_min = 0
    aim_path_blocks = []
    for block in path_set[0]['blocks']:
        aim_path_blocks.append(block)
    for block in path_set[0]['blocks']:
        for other_block in blocks:
            if other_block not in aim_path_blocks:
                data = get_min_path(block['nodes'], other_block['nodes'])
                d_temp = data[2]
                if d_min == 0 and len(path_temp) == 0:
                    d_min = d_temp
                    path_temp = data[1]
                else:
                    if d_temp < d_min:
                        d_min = d_temp
                        path_temp = data[1]
    if len(path_temp) != 0:
        path = path_temp
        cost = d_min  # d_min其实就是连通代价
    return path, cost, path_set


def get_min_path(b1, b2):  # b1, b2当前要判断的点集
    block = []
    d_min = 0
    link = []
    for node in b1:
        x = node['x']
        y = node['y']
        for anode in b2:
            d = (anode['x'] - x) ** 2 + (anode['y'] - y) ** 2
            if d_min == 0:
                d_min = d
                block = [node['block'], anode['block']]
                link = [node, anode]
            else:
                if d < d_min:
                    d_min = d
                    block = [node['block'], anode['block']]
                    link = [node, anode]
    return block, link, np.sqrt(d_min)


''' 还是当我这一下午啥也没写吧。。。。。
def get_path_set(paths):
    conn_to_sink = []
    path_sets = []
    for path in paths:
        if path[1] == sink:
            conn_to_sink.append(path)
            path[1]['block'] = 'sink'
    path_all = paths[:]
    print('直连sink 的路线', len(conn_to_sink), conn_to_sink)
    path_remain = []
    while len(conn_to_sink) != 0:
        path_set = ['sink']
        block_connected = []
        start = 'sink'
        for path in path_all:
            if path[1]['block'] == start or start:
                start = True
                if path[1]['block'] in path_set or path[0]['block'] in path_set:
                    if path[1]['block'] in path_set:
                        block = path[0]['block']
                        if block not in block_connected:
                            not_in_pr = True
                            for pr in path_remain:
                                if block in pr:
                                    path_set.extend(pr)
                                    block_connected.extend(pr)
                                    path_remain.remove(pr)
                                    not_in_pr = False
                                    break
                                else:
                                    not_in_pr = True
                            if not_in_pr:
                                path_set.append(block)
                                block_connected.append(block)
                                path_all.remove(path)
                    elif path[0]['block'] in path_set:
                        block = path[1]['block']
                        if block not in block_connected:
                            not_in_pr = True
                            for pr in path_remain:
                                if block in pr:
                                    path_set.extend(pr)
                                    block_connected.extend(pr)
                                    path_remain.remove(pr)
                                    not_in_pr = False
                                    break
                                else:
                                    not_in_pr = True
                            if not_in_pr:
                                path_set.append(block)
                                block_connected.append(block)
                                path_all.remove(path)
                else:
                    # 生成很多剩余点的集合，已遍历，但目前不属于任何一个区域，需要进一步区分
                    exist = True
                    print(path[0]['block'], path[1]['block'])
                    if len(path_remain) != 0:
                        for pr in path_remain:
                            print(pr)
                            if path[0]['block'] not in pr and path[1]['block'] not in pr:
                                exist = False
                                continue
                            else:
                                exist = True
                                if path[0]['block'] not in pr:
                                    pr.append(path[0]['block'])
                                if path[1]['block'] not in pr:
                                    pr.append(path[1]['block'])
                                break
                        if not exist:
                            path_remain.append([path[0]['block'], path[1]['block']])
                    else:
                        path_remain.append([path[0]['block'], path[1]['block']])
                    path_all.remove(path)
                    print('path_remain', path_remain)
            print('path:', path_set)
        del conn_to_sink[0]
        if len(path_set) != 1:
            path_sets.append({'path': path_set, 'load': len(path_set)-1})
            print('生成的path:', path_set)
    return path_sets'''


def destroy(s, dp):
    sr = []
    s1 = s[:]
    delete = int(dp*len(s))  # 被破坏点的个数
    CH = random.sample(range(0,len(s)-1), delete)  # 在规定范围中产生不同的随机数
    for i in CH:
        s1[i] = {'destroy': True}
    for node in s1:
        if 'destroy' in node.keys():
            continue
        else:
            sr.append(node)
    # print('原节点个数:', len(s),'删除节点个数：', delete,'剩余节点个数：', len(sr))
    return sr

'''
def get_node_conn_2021(s, cl, R, sink):
    num = 0
    for index in range(cl):
        aim_block = []
        for node in s:
            if node['block'] == index:
                aim_block.append(node)
        num = num + get_node_conn(aim_block, R, sink)
    return num
'''


def get_node_conn(s, R, sink, paths):
    # 返回一个点集中仍可连接的点个数
    s1 = s[:]
    for node in s1:
        node['block'] = -1
    sr = []
    sr.append(sink)
    i = 0
    while i < len(sr):
        aim = sr[i]
        for index, node in enumerate(s):
            if node not in sr:
                d = (node['x'] - aim['x']) ** 2 + (node['y'] - aim['y']) ** 2
                if d < R ** 2:
                    sr.append(node)
                else:
                    for path in paths:
                        if (path[0] == node and path[1] == aim) or (path[1] == node and path[0] == aim):
                            sr.append(node)
        i = i + 1
    sr.remove(sink)
    '''
    for id, ubn in enumerate(s1):
        if ubn['block'] == -1:  # 选出未被连接的
            sr = []
            sr.append(ubn)
            i = 0
            while i < len(sr):
                aim = sr[i]
                for index, node in enumerate(s):
                    if node not in sr:
                        d = (node['x']-aim['x'])**2 + (node['y']-aim['y'])**2
                        if d < R**2:
                            sr.append(node)
                            s1[index]['block'] = id
                        else:
                            for path in paths:
                                if (path[0] == node and path[1] == aim) or (path[1] == node and path[0] == aim):
                                    sr.append(node)
                                    s1[index]['block'] = id
                i = i+1
            if len(sr) > max_len:
                max_len = len(sr)
    '''
    return len(sr), sr


def load_balance(paths, conn_path):
    # 输入path, 和step3的连接路径conn_path
    # 输出方差值
    load = []
    conn = [-1,-1]
    sum = 0
    if len(conn_path) != 0:
        for path_set in paths:
            for block in path_set['blocks']:
                if conn_path[0] in block['nodes']:
                    conn[0] = path_set['id']
                    for b in path_set['blocks']:
                        sum = len(b['nodes']) + sum
                if conn_path[1] in block['nodes']:
                    conn[1] = path_set['id']
                    for b in path_set['blocks']:
                        sum = len(b['nodes']) + sum
        av_load = sum/len(conn)
    for path_set in paths:
        load_temp = 0
        if path_set['id'] == conn[1] or path_set['id'] == conn[0]:
            load.append(av_load)
        else:
            for block in path_set['blocks']:
                load_temp = len(block['nodes']) + load_temp
            load.append(load_temp)
    v = np.var(load)
    print('The Load of Each Road:', load, '\nVariance:', v)
    return load


def network_lifetime(load, E, R):
    load_max = max(load)  # 用于计算网络寿命
    et = load_max*(50/pow(10, 9) + 10/pow(10, 12)*(R**2))
    er = load_max*(50/pow(10, 9))
    t = E/(et+er)
    print('The lifetime of this network is:', t)


# PARAMETERS ##############################
xm = 1000    # 横坐标长度
ym = 1000   # 纵坐标长度

# 基站定义
sink = {}
sink['x'] = xm/2  # 基站横坐标
sink['y'] = ym-50  # 基站纵坐标

# N = 16   每个区域的节点个数
R = 50  # 节点通信半径
[w, h] = [25, 25]  # 网格画图长宽，实际宽度为25
D = 2*xm/3   # 近距离块的划分
Dm = 0.71*(xm - w)   # 理论上两格之间的最长距离

a1 = 0.218            # 权重a1 ---> 连接成本
a2 = 1-a1             # a2 ----> 健壮性和负载均衡
Beita1 = 0.25          # 权重β1 ---> 健壮性
Beita2 = 1 - Beita1   # β2 ---> 连接成本

Dp = 0.5      # 随机破坏节点比例
dg = 1     # 每个结点产生数据的速率 1bit/round (可以直接用结点数表示)
E = 0.5    # 每个结点的满电能量
# END OF PARAMETERS ########################


# 人为指定中心点 ###########################
C_20 = [(50, 100), (100, 400), (50, 700), (30, 950), (300, 30), (340, 350), (260, 680), (380, 940),
        (500, 200), (500, 550), (590, 720), (520, 900), (730, 100), (600, 400), (780, 830),
        (950, 30), (840, 300), (870, 500), (980, 700), (950, 950)]
C_20 = to_obj(C_20)

C_15 = [(50, 100), (100, 400), (50, 700), (50, 900), (340, 340), (400, 660),
        (500, 950), (500, 200), (590, 720), (730, 100), (650, 480), (780, 830), (890, 30), (870, 400), (900, 950)]
C_15 = to_obj(C_15)


# 当区域数为15时 ###########################
# 建立一个块的字典数组，主要关注块的操作，但同时还有点的操作，可以把点的数据存在块的某个属性中
S_15, block_1, block_15 = create_nodes(C_15, 30, R, xm, ym)
S = S_15[:]
print(len(S))

print('Mobile Relay Algorithm:')
# 2021 移动小车网络：健壮性、负载均衡性
# S_15_grid = to_grid(S_15, w,xm, ym)
# for blo in block_15:
#   blo['nodes'] = to_grid(blo['nodes'], w,xm, ym)
B_15, S = get_border(S, R)  # 边界点
S, conn_sink, block_15 = get_distance_to_sink(S, len(C_20), block_15, sink)
S, conn_segm, block_15 = get_distance_to_segm(S, len(C_20), block_15)
path_2021, cost_2021_1 = step1_2021(block_15, D, a1, a2, R, sink)
print(len(S))
block_15 = get_plumpness(block_15, xm,ym, w, h)
path_2021_2, cost_2021_2 = step2_2021(block_15, D, Dm, Beita1, Beita2)
path_2021.extend(path_2021_2)
path_2021_3, cost_2021_3, path_set_2021 = step3_2021(block_15)
# 实验三
load_2021 = load_balance(path_set_2021, path_2021_3)
network_lifetime(load_2021, E, R)
if len(path_2021_3) != 0:
    path_2021.append(path_2021_3)
cost_2021 = cost_2021_1 + cost_2021_2
# print('All cost:', cost_2021, 'Costs each step：', cost_2021_1, cost_2021_2,cost_2021_3)

'''EXP2
sum = 0
for i in range(100):
    S2 = destroy(S_15, Dp)
    lcn_2021, lcn = get_node_conn(S2, R, sink, path_2021)
    exist_rate_2021 = lcn_2021/len(S2)
    sum = sum + exist_rate_2021
er_average = sum/100
print('Mobile Relay Algorithm Existing Rate:', er_average)
'''
# 作图  ####################################
gdf = make_mesh([0, 0, xm, ym], w, h)
gdf.boundary.plot()
draw_nodes(S_15)
'''
draw_nodes(S2)
for i in lcn:
    plt.plot(i['x'], i['y'], 'mv')
'''
draw_line(path_2021)
plt.plot(sink['x'], sink['y'], 'rp')  # 绘制sink点
plt.annotate('sink', xy=(sink['x'], sink['y']), xytext=(-20, 10),
             textcoords='offset points', fontsize=12, color='r')

plt.show()
''''''
