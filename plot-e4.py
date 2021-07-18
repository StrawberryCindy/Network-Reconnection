import matplotlib.pyplot as plt


# 画图 #########################
x = [16,18,20,22,24,26,28,30]

# 实验四
# 区域数为15
y11 = [25000,18020,15832,17623,15932,15325,15234,12117]
y21 = [20833,20202,16666,15151,15151,19230,14285,10256]
y31 = [38461,29629,40000,36363,27777,25641,19047,17777]
y12 = [20833,20232,19829,17700,15532,15151,12325,11251]
y22 = [22727,18518,18181,20202,15151,12820,15904,12121]
y32 = [41667,44445,26666,24242,30303,20512,25367,17777]
'''
# 区域数为20
y11 = [17000,13202,13333, 9623,12820,11904, 8928, 8888]
y21 = [17857,12345,12500,11363,11904,11111,11834, 9538]
y31 = [29411,26143,22222,20202,16666,17094,15875,13333]
y12 = [10833,12232,11111,15700,11904,11523,13325,11251]
y22 = [16666,15873,13333,14986,13986,13986,10989,12121]
y32 = [25000,24777,20000,18181,19607,15384,15038,14815]
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

figsize = 4.5,4
figure, ax = plt.subplots(figsize=figsize)
plt.plot(x, y1, color='mediumseagreen', label='MRSC', Marker='^')
plt.plot(x, y2, color='cornflowerblue', label='RTN', linestyle='-', Marker='o')
plt.plot(x, y3, color='lightsalmon', label='RDNR', linestyle='-', Marker='s')


# 实验四
# plt.title("Comparison of the lifetime")
plt.xlabel("Maximum number of grids of a segment")
plt.ylabel("Network longevity ($10^{3}$round)")
plt.axis([15, 31, 10, 45])  #6, 36
plt.legend(loc="upper left")  # lower   right

plt.show()

