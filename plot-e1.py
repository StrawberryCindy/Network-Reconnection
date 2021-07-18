import matplotlib.pyplot as plt


# 画图 #########################
x = [16,18,20,22,24,26,28,30]

# 实验一
# 区域数为15
y11 = [5005,3830,1878,1651,1220,1496,802,2117]
y21 = [6800,6921,6214,6148,5790,5724,5173,5517]
y31 = [2008,1972,1745,1595,1671,1432,1295,1244]
y12 = [2399,1906,2953,2783,1700,1041,990,738]
y22 = [6372,6109,6584,6090,5794,4694,5064,4024]
y32 = [2132,1904,1700,1628,1434,1566,1306,1509]
'''
# 区域数为20
y11 = [1172, 844,1263, 815, 578,1212, 868, 465]
y21 = [7117,7937,7444,7264,6353,7186,6290,5987]
y31 = [2301,1859,1966,1948,1502,1436,1227, 945]
y12 = [2605,1848,1469,1938,1933,1363, 783, 603]
y22 = [8327,7548,7527,7264,6949,6041,6088,5787]
y32 = [2114,2206,1841,1731,1638,1567,1233, 957]
'''

y1 = []
for i, y in enumerate(y11):
    y1.append(((y+y12[i])/2)/10**3)
y2 = []
for i, y in enumerate(y21):
    y2.append(((y+y22[i])/2)/10**3)
y3 = []
for i, y in enumerate(y31):
    y3.append(((y+y32[i])/2)/10**3)

figsize = 5.19,4
figure, ax = plt.subplots(figsize=figsize)
plt.plot(x, y1, color='mediumseagreen', label='MRSC', Marker='^')
plt.plot(x, y2, color='cornflowerblue', label='RTN', linestyle='-', Marker='o')
plt.plot(x, y3, color='lightsalmon', label='RDNR', linestyle='-', Marker='s')

# 实验一
# plt.title("Comparison of moving cost")
# 设置横纵坐标的名称以及对应字体格式
font1 = {
         'weight': 'normal',
         'size': 14,
         }
# 设置坐标轴的粗细
ax = plt.gca()  # 获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(1)  # 设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(1)  # 设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(1)  # 设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(1)  # 设置上部坐标轴的粗细
plt.xlabel("Maximum number of grids of a segment", font1)
plt.ylabel("Total length of the path ($10^{3}$m)", font1)
plt.axis([15.5, 30.5, 0, 8])
plt.tick_params(labelsize=12)
plt.legend(bbox_to_anchor=(0.03, 0.49), loc=3, borderaxespad=0,
           edgecolor='k', fancybox=False,
           fontsize=13)
plt.tight_layout() #去除pdf周围白边
plt.savefig('test.png',bbox_inches='tight',dpi=figure.dpi,pad_inches=0.0)

plt.show()

