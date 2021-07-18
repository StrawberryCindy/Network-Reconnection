import matplotlib.pyplot as plt


# 画图 #########################

# 实验二
x = [10, 15, 20, 25, 30, 35, 40]
'''
# block15, 18
y1 = [13.5, 12, 10, 6, 5, 1.2, 1]
y2 = [20, 15.3, 9.4, 7, 5.5, 4.67, 3]
y3 = [54.5, 41.7, 34.9, 29.3, 25.8, 20.7, 18.4]


# block15，28
y1 = [22.4, 14, 10.1, 7.62, 5.5, 3.7, 2]
y2 = [20.04, 15.69, 9.9, 7.5, 6, 5, 4]
y3 = [58.3, 44.7, 38.4, 31, 25.5, 20.8, 15.2]
'''
'''
# block20，18
y1 = [22, 16, 8, 6, 7.13 ,6.2, 4]
y2 = [21, 19, 12, 7.4, 5.5, 4.4, 1.4]
y3 = [59.8, 42.2, 36.7, 29.8, 26.8, 20.4, 18.5]
'''
# block20，28
y1 = [22, 20, 18, 13, 8, 7.5, 3.5]
y2 = [23, 20, 13.3, 8, 7, 6.7, 4.1]
y3 = [60.6, 54.6, 45, 34, 26.4, 22 ,19.5]


figsize = 4.5,4.2
figure, ax = plt.subplots(figsize=figsize)
plt.plot(x, y1, color='mediumseagreen', label='MRSC', Marker='^')
plt.plot(x, y2, color='cornflowerblue', label='RTN', linestyle='-', Marker='o')
plt.plot(x, y3, color='lightsalmon', label='RDNR', linestyle='-', Marker='s')

# 实验二
# plt.title("Comparison of Alive Nodes After Second Destroy")
plt.xlabel("Percentage of destroyed nodes")
plt.ylabel("Percentage of alive nodes")
# plt.grid(b=True, axis='y')  # 只显示y轴网格线

ax=plt.gca()  # 获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(1)  # 设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(1)  # 设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(1)  # 设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(1)  # 设置上部坐标轴的粗细
plt.axis([9, 41, -5, 65])
plt.legend(bbox_to_anchor=(0.03, 0.47), loc=3, borderaxespad=0,
           edgecolor='k', fancybox=False)


plt.show()

