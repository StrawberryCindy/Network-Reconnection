import RoundTable
import mrsc
import MobileRelay
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


# PARAMETERS ##############################
xm = 1000    # 横坐标长度
ym = 1000   # 纵坐标长度

# 基站定义
sink = {}
sink['x'] = xm/2  # 基站横坐标
sink['y'] = ym-50  # 基站纵坐标

# N = 16    每个区域的节点个数
R = 50  # 节点通信半径
[w, h] = [25, 25]  # 网格画图长宽，实际宽度为25
D = 2*xm/3   # 近距离块的划分
Dm = 0.71*(xm - w)   # 理论上两格之间的最长距离

a1 = 0.2              # 权重a1 ---> 连接成本
a2 = 1-a1             # a2 ----> 健壮性和负载均衡
Beita1 = 0.2          # 权重β1 ---> 健壮性
Beita2 = 1 - Beita1   # β2 ---> 连接成本
# END OF PARAMETERS ########################

# 人为指定中心点 ###########################
C_20 = [(50, 100), (100, 400), (50, 700), (30, 950), (300, 30), (340, 350), (260, 680), (280, 890),
        (500, 200), (500, 550), (590, 720), (570, 900), (730, 100), (600, 400), (780, 830),
        (950, 30), (840, 300), (870, 500), (980, 700), (950, 950)]
C_20 = to_obj(C_20)

C_15 = [(50, 100), (100, 400), (50, 700), (50, 900), (340, 340), (400, 660),
        (500, 950), (500, 200), (590, 720), (730, 100), (650, 480), (780, 830), (890, 30), (870, 400), (900, 950)]
C_15 = to_obj(C_15)


def main(c, mn, Dp):
    print(len(c), mn,'破坏比例：',Dp)
    # 当区域数为20时 ###########################
    S, block_1, block_2 = MobileRelay.create_nodes(c, mn, R, xm, ym)
    print(block_2)

    '''
    # 2016 移动中继相关
    B_2016, S_2016 = mrsc.get_border(S, R)  # 边界点
    S_2016 = mrsc.get_move_2016(S, block_1, 9, mn)
    M = []
    for node in S_2016:
        if node['movable']:
            M.append(node)
    conn_path_2016, cost_2016, move_path_2016, DN_2016 = mrsc.get_min_path_2016(B_2016, M, R)
    if cost_2016 == 0:
        print('没有找到可行的路径')
    else:
        pass
        S1_2016 = S[:]
        for dn in DN_2016:
            S1_2016.append(dn)
        S2_2016 = mrsc.destroy(S1_2016, Dp)
        lcn_2016 =  mrsc.get_node_conn(S2_2016, R, sink)
        exist_rate_2016 = lcn_2016/len(S1_2016)
        print('MRSC Algorithm Existing Rate:', exist_rate_2016)
    '''
    # 2018 圆桌协议相关
    B_2018, S_2018 = RoundTable.get_border(S, R)  # 边界点
    B_sorted_2018 = RoundTable.sort_border(B_2018, len(c))  # 按区域划分的二维边界点集合
    block_2018, d_cost_2018_1 = RoundTable.get_d_2018(B_sorted_2018, R, xm, ym)
    conn_block_2018, conn_path_2018, conn_node_2018 = RoundTable.get_min_path_2018(B_2018, R)
    S_2018 = RoundTable.get_relay_2018(S_2018, conn_block_2018, conn_node_2018)
    DN_2018 = RoundTable.desired_node_location_2018(conn_node_2018, R)
    d_cost_2018_2, move_path_2018 = RoundTable.get_replace_cost(DN_2018, S_2018, R)

    S1_2018 = S_2018[:]
    for dn in DN_2018:
        S1_2018.append(dn)
    S2_2018 = RoundTable.destroy(S1_2018, Dp)
    lcn_2018 = RoundTable.get_node_conn(S2_2018, R, sink)
    exist_rate_2018 = lcn_2018 / len(S1_2018)
    print('RoundTable Algorithm Existing Rate:', exist_rate_2018)

    # 2021 移动小车网络：健壮性、负载均衡性
    S_grid = MobileRelay.to_grid(S, w, xm, ym)
    for blo in block_2:
        blo['nodes'] = MobileRelay.to_grid(blo['nodes'], w,xm, ym)
    B_2021, S_2021 = MobileRelay.get_border(S_grid, R)  # 边界点
    S_2021, conn_sink, block_2021 = MobileRelay.get_distance_to_sink(S_2021, len(c), block_2, sink)
    S_2021, conn_segm, block_2021 = MobileRelay.get_distance_to_segm(S_2021, len(c), block_2021)
    print(block_2021)
    path_2021, cost_2021_1 = MobileRelay.step1_2021(block_2021, D, a1, a2, R, sink)

    block_2021 = MobileRelay.get_plumpness(block_2021,xm, ym, w, h)
    path_2021_2, cost_2021_2 = MobileRelay.step2_2021(block_2021, D, Dm, Beita1, Beita2)
    path_2021.extend(path_2021_2)
    path_2021_3, cost_2021_3, path_set_2021 = MobileRelay.step3_2021(block_2021)
    path_2021.append(path_2021_3)

    S2_2021 = MobileRelay.destroy(S, Dp)
    lcn_2021 = MobileRelay.get_node_conn(S2_2021, R, sink, path_2021)
    exist_rate_2021 = lcn_2021 / len(S)
    print('Mobile Relay Algorithm Existing Rate:', exist_rate_2021)
    return exist_rate_2016, exist_rate_2018, exist_rate_2021

'''
# 画图 #########################
x = [16,18,20,22,24,26,28,30]
y11 = [5005,3830,1878,1651,1220,1496,802,2117]
y21 = [6800,6921,6214,6148,5790,5724,5173,5517]
y31 = [3351,3229,2786,2438,2276,2238,2146,2014]
y12 = [2399,1906,2953,2783,1700,1041,990,738]
y22 = [6372,6109,6584,6090,5794,4694,5064,4024]
y32 = [2803,2969,2705,2180,1937,1998,2093,2100]

data = 
y11[4] = data[0]
y21[4] = data[1]
y31[4] = data[2]

y1 = []
for i, y in enumerate(y11):
    y1.append((y+y12[i])/2)
y2 = []
for i, y in enumerate(y21):
    y2.append((y+y22[i])/2)
y3 = []
for i, y in enumerate(y31):
    y3.append((y+y32[i])/2)
'''
''''''
x = [0.1, 0.15 ,0.2, 0.25, 0.3]
y_temp = [[]*5 for i in range(3)]
for i in range(3):
    y_temp[i][0] = main(C_15, 18, 0.1)
    y_temp[i][1] = main(C_15, 18, 0.15)
    y_temp[i][2] = main(C_15, 18, 0.2)
    y_temp[i][3] = main(C_15, 18, 0.25)
    y_temp[i][4] = main(C_15, 18, 0.3)

    '''  
    main(C_15, 28, 0.1)
    main(C_15, 28, 0.15)
    main(C_15, 28, 0.2)
    main(C_15, 28, 0.25)
    main(C_15, 28, 0.3)
    main(C_20, 18, 0.1)
    main(C_20, 18, 0.15)
    main(C_20, 18, 0.2)
    main(C_20, 18, 0.25)
    main(C_20, 18, 0.3)
    
    main(C_20, 28, 0.1)
    main(C_20, 28, 0.15)
    main(C_20, 28, 0.2)
    main(C_20, 28, 0.25)
    main(C_20, 28, 0.3)
    '''
y1 = []*5
y2 = []*5
y3 = []*5
for i in range(5):
    y1[i] = (y_temp[0][i][0] + y_temp[1][i][0] + y_temp[2][i][0])/3
    y2[i] = (y_temp[0][i][1] + y_temp[1][i][1] + y_temp[2][i][1])/3
    y3[i] = (y_temp[0][i][2] + y_temp[1][i][2] + y_temp[2][i][2])/3
'''
main(C_15, 26)
main(C_15, 28)
main(C_15, 30)
'''
plt.figure(1)
plt.plot(x, y1, color='lightsalmon', label='MRSC', Marker='s')
plt.plot(x, y2, color='cornflowerblue', label='RoundTable', linestyle='-', Marker='^')
plt.plot(x, y3, color='mediumseagreen', label='Mobile Relay', linestyle='-', Marker='o')

plt.title("Comparison of Existing Node Rate After Second Destroy")
plt.xlabel("Destroy Rate")
plt.ylabel("Existing Node Rate")
plt.axis([0, 0.4, 0, 1])
plt.grid(b=True, axis='y')  # 只显示y轴网格线
plt.legend(loc="upper right")  # lower   right
plt.show()

