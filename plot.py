import matplotlib.pyplot as plt


# 画图 #########################
x = [16,18,20,22,24,26,28,30]
# 区域数为15

'''
y11 = [4096,5184,10000,3651,6220,8496,8902,12117]
y21 = [6400,8100,7744,9787,4176,13670,7056,8600]
y31 = [ 405, 729,1280,1301,1152,3506,4000, 4120]
y12 = [4300,3296,2953,6783,7700,7041,3990, 2738]
y22 = [4096,5184,7744,3936,7584, 676,7056,8200]
y32 = [ 656, 546,  0,   0, 1200,   0,  0, 0]
'''
# 实验四
# 区域数为15
y11 = [25000,18020,15832,17623,15932,15325,15234,12117]
y21 = [20833,20202,16666,15151,15151,19230,14285,10256]
y31 = [38461,29629,40000,36363,27777,25641,19047,17777]
y12 = [20833,20232,19829,17700,15532,14523,12325,11251]
y22 = [22727,18518,18181,20202,15151,12820,15904,12121]
y32 = [41667,44445,26666,24242,30303,20512,25367,17777]
y1 = []
for i, y in enumerate(y11):
    y1.append((y+y12[i])/2)
y2 = []
for i, y in enumerate(y21):
    y2.append((y+y22[i])/2)
y3 = []
for i, y in enumerate(y31):
    y3.append((y+y32[i])/2)

plt.figure(1)
plt.plot(x, y1, color='lightsalmon', label='MRSC', Marker='s')
plt.plot(x, y2, color='cornflowerblue', label='RoundTable', linestyle='-', Marker='^')
plt.plot(x, y3, color='mediumseagreen', label='Mobile Relay', linestyle='-', Marker='o')
'''实验一
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

'''

'''
# 实验二
# block15, 18
x = [0.1,0.15,0.2,0.25,0.3]
y1 = [0.22, 0.2, 0.18, 0.13, 0.08]
y2 = [0.15, 0.1, 0.09, 0.07, 0.06]
y3 = [0.335, 0.257, 0.23, 0.18, 0.16]
# block15，28
y1 = [0.22, 0.2, 0.18, 0.13, 0.08]
y2 = [0.15, 0.1, 0.09, 0.07, 0.06]
y3 = [0.41, 0.4, 0.32, 0.24, 0.16]

'''
'''实验一
plt.title("Comparison of moving cost")
plt.xlabel("The maximum number of grids of a segment")
plt.ylabel("Moving cost")
plt.axis([14, 32, 0, 9000])
plt.grid(b=True, axis='y')  # 只显示y轴网格线
plt.legend(loc="upper right")  # lower   right
plt.show()
'''
'''实验二
plt.title("Comparison of Existing Node Rate After Second Destroy")
plt.xlabel("Destroy Rate")
plt.ylabel("Existing Node Rate")
plt.axis([0.05, 0.35, 0, 0.4])
plt.grid(b=True, axis='y')  # 只显示y轴网格线
plt.legend(loc="upper right")  # lower   right
plt.show()
'''
#实验三
plt.title("Load Balancing Degree")
plt.xlabel("The maximum number of grids of a segment")
# plt.ylabel("The variance of network traffic")
# plt.axis([14, 32, 0, 10000])
plt.ylabel("The lifetime of this network")
plt.axis([14, 32, 5000, 45000])
plt.grid(b=True, axis='y')  # 只显示y轴网格线
plt.legend(loc="upper right")  # lower   right
plt.show()
