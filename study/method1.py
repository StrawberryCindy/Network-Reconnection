import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from operator import itemgetter


xm = 100    # 横坐标长度
ym = 100   # 纵坐标长度
sink = {'x': 0, 'y': 0}   # 基站定义
sink['x'] = xm/2  # 基站横坐标
sink['y'] = ym-10  # 基站纵坐标
density = 3/10   # 节点密度
n = int(xm*ym*density)   # 节点个数
R = 5   # 节点通信半径
[w, h] = [5, 5]  # 网格长宽
# END OF PARAMETERS ########################

S = []  # 节点集
for i in range(0, n):
    S.append({'xd': random.random()*xm, 'yd': random.random()*ym, 'type': 'N'})
    # xy坐标 type:N普通节点

S.append({'xd': sink['x'], 'yd': sink['y'], 'type': 'S'})  # sink也是一个节点
plt.figure(1)
plt.subplot(1, 2, 1)
for i in range(0, n):
    plt.plot(S[i]['xd'], S[i]['yd'], 'k.')

plt.plot(S[n]['xd'], S[n]['yd'], 'rp')  # 绘制sink点
plt.annotate('sink', xy=(S[n]['xd'], S[n]['yd']), xytext=(-20, 10),
             textcoords='offset points', fontsize=12, color='r')

plt.ylabel('Y-axis (m)', fontsize=10)
plt.xlabel('X-axis (m)', fontsize=10)
plt.axis([0, 100, 0, 100])  # 设置图的大小
plt.axis('equal')   # 图横纵坐标尺相同
plt.title('Initial Node')

# 破坏一些点 ####################################
'''随机破坏（效果不理想）
Unique_num = n/2  # 被破坏点集的个数
CH = random.sample(range(0, n), Unique_num)  # 在规定范围中产生不同的随机数
for i in CH:
    i = int(i)
    S2 = S[:]
    node = S[i]
    del S2[i]
    flag = 0
    for index, other_node in enumerate(S2):
        flag = flag+1
        if flag > 10:
            continue
        else:
            if np.abs(other_node['xd'] - node['xd']) < 2 * R and np.abs(other_node['yd'] - node['yd']) < 2 * R:
                CH.append(index)
            else:
                continue
CH.sort()'''

# 指定范围破坏点 ################
CH = []
for index, node in enumerate(S):
    x = node['xd']
    y = node['yd']
    if 0.2*xm < x < 0.3*xm or 0.7*xm < x < 0.8*xm or 0.2*ym < y < 0.3*ym or (0<x<0.1*xm and 0<y<0.2*ym) or 0.7*ym < y < 0.8*ym:
        CH.append(index)

temp_x = []
temp_y = []
S1 = S[:]  # 存放经破坏后当前存在的所有点,切片赋值，使指针指向不同的空间
for i in reversed(CH):
    temp_x.append(S[i]['xd'])
    temp_y.append(S[i]['yd'])
    del S1[i]

Unique_node_x = temp_x
Unique_node_y = temp_y


# 判断边界点 ###################
B = []  # 用来存放边界点们
for index, node in enumerate(S1):
    angle_r_all = []
    right_max = 0
    S2 = S1[:]
    del S2[index]  # 中间变量，存放除本节点外的其他点
    for other_node in S2:
        if np.abs(other_node['xd'] - node['xd']) < 2*R and np.abs(other_node['yd'] - node['yd']) < 2*R:
            d = np.sqrt((node['xd'] - other_node['xd']) ** 2 + (node['yd'] - other_node['yd']) ** 2)
            if d < 2*R:
                angle_d = np.arccos(((d/2)/R))
                if node['xd'] > other_node['xd']:
                    center_angle = np.arctan((other_node['yd']-node['yd'])/(other_node['xd']-node['xd'])) + np.pi
                elif node['xd'] == other_node['xd']:
                    if node['yd'] > other_node['yd']:
                        center_angle = -np.pi/2
                    else:
                        center_angle = np.pi/2
                else:
                    center_angle = np.arctan((other_node['yd']-node['yd'])/(other_node['xd']-node['xd']))
                # 左右边界
                left = center_angle - angle_d
                right = center_angle + angle_d
                angle_r = [left, right]
                angle_r_all.append(angle_r)

    angle_r_all = sorted(angle_r_all, key=itemgetter(0))  # 按left升序排序
    if angle_r_all[0][0] >= -np.pi / 2:
        node['type'] = "B"  # 不覆盖角度-π/2，判断为边界节点
        B.append(node)
        continue
    for i in range(0, len(angle_r_all)):
        if angle_r_all[i][0] > right_max:  # 当左端值不与右端当前最大值发生重合时，判为边界点
            node['type'] = "B"
            B.append(node)
            break
        if angle_r_all[i][1] > right_max:  # 如果此右端点大于右端当前最大值，进行迭代
            right_max = angle_r_all[i][1]
    if right_max < np.pi*3/2:
        node['type'] = "B"  # 未覆盖到全角度，判断为边界点
        B.append(node)

plt.subplot(1, 2, 2)
for i in range(0, n - len(CH)):
    plt.plot(S1[i]['xd'], S1[i]['yd'], 'k.')

for i in range(0, len(B)):
    plt.plot(B[i]['xd'], B[i]['yd'], 'gd')

# 画边界连线
'''
for index, node in enumerate(B):
    B2 = B[:]
    del B2[index]
    for other_node in B2:
        if np.abs(other_node['xd'] - node['xd']) < R and np.abs(other_node['yd'] - node['yd']) < 2*R:
            d = np.sqrt((node['xd'] - other_node['xd']) ** 2 + (node['yd'] - other_node['yd']) ** 2)
            if d < 2*R:
                plt.plot([node['xd'], other_node['xd']], [node['yd'], other_node['yd']], color='g')
'''

print('初始节点总数：', len(S), '被破坏后的剩余节点数：', len(S1))
plt.plot(S[n]['xd'], S[n]['yd'], 'r*')  # 绘制sink点
plt.annotate('sink', xy=(S[n]['xd'], S[n]['yd']), xytext=(-20, 10),
             textcoords='offset points', fontsize=12, color='r')
plt.ylabel('Y-axis (m)', fontsize=10)
plt.xlabel('X-axis (m)', fontsize=10)
plt.axis([0, 100, 0, 100])  # 设置图的大小
plt.axis('equal')
plt.title('After Being Destroyed')

plt.show()   # 展示图像

