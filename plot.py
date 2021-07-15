import matplotlib.pyplot as plt


# 画图 #########################
'''实验一

x = [16,18,20,22,24,26,28,30]

# 区域数为15
y11 = [5005,3830,1878,1651,1220,1496,802,2117]
y21 = [6800,6921,6214,6148,5790,5724,5173,5517]
y31 = [2008,1972,1745,1595,1671,1432,1295,1244]
y12 = [2399,1906,2953,2783,1700,1041,990,738]
y22 = [6372,6109,6584,6090,5794,4694,5064,4024]
y32 = [2132,1904,1700,1628,1434,1566,1306,1509]

区域数为20
y11 = [1172, 844,1263, 815, 578,1212, 868, 465]
y21 = [7117,7937,7444,7264,6353,7186,6290,5987]
y31 = [2301,1859,1966,1948,1502,1436,1227, 945]
y12 = [2605,1848,1469,1938,1933,1363, 783, 603]
y22 = [8327,7548,7527,7264,6949,6041,6088,5787]
y32 = [2114,2206,1841,1731,1638,1567,1233, 957]

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

# 实验二
x = [0.1,0.15,0.2,0.25,0.3]
y1 = [0.22, 0.15, 0.16,]
y2 = [0.15, 0.1, 0.09, 0.07, 0.06]
y3 = [0.335, 0.257, 0.206, 0.17, 0.158]

plt.figure(1)
plt.plot(x, y1, color='lightsalmon', label='MRSC', Marker='s')
plt.plot(x, y2, color='cornflowerblue', label='RoundTable', linestyle='-', Marker='^')
plt.plot(x, y3, color='mediumseagreen', label='Mobile Relay', linestyle='-', Marker='o')

'''实验一
plt.title("Comparison of moving cost")
plt.xlabel("Maximum number of nodes in each block")
plt.ylabel("Moving cost")
plt.axis([14, 32, 0, 9000])
plt.grid(b=True, axis='y')  # 只显示y轴网格线
plt.legend(loc="upper right")  # lower   right
plt.show()
'''