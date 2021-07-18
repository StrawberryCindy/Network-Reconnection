import matplotlib.pyplot as plt


# 画图 #########################
x = [16,18,20,22,24,26,28,30]

# 实验三
# 区域数为15
y11 = [4096,5184,10000,3651,6220,8496,8902,12117]
y21 = [6400,8100,7744,9787,4176,13670,7056,8600]
y31 = [ 405, 729,1280,1301,1152,3506,4000, 4120]
y12 = [4300,3296,2953,6783,7700,7041,3990, 2738]
y22 = [4096,5184,7744,3936,7584, 676,7056,8200]
y32 = [ 656, 546,  0,   0, 1200,   0,  0, 0]
'''
# 区域数为20
y11 = [4096,15584,16900,13651,14336,11664,33124,27225]
y21 = [5184,23409,12100,17424,32400,22016,15427,26600]
y31 = [3456,4374,6400,5270,6272,10816,12544,14450]
y12 = [14300,7296,15580,16783,11664,16041,   0, 3600]
y22 = [17744,6561,15288,15499,2414,7811,19979, 8200]
y32 = [ 0,1333,     0,1721,3872,   0,  0,    0]

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

figsize = 4.5,4.2
figure, ax = plt.subplots(figsize=figsize)
plt.plot(x, y1, color='mediumseagreen', label='MRSC', Marker='^')
plt.plot(x, y2, color='cornflowerblue', label='RTN', linestyle='-', Marker='o')
plt.plot(x, y3, color='lightsalmon', label='RDNR', linestyle='-', Marker='s')

# 实验三
plt.xlabel("Maximum number of grids of a segment")
plt.ylabel("Deviation of data traffic ($10^{3})")
# 设置坐标轴的粗细
ax=plt.gca()  # 获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(1)  # 设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(1)  # 设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(1)  # 设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(1)  # 设置上部坐标轴的粗细
plt.axis([15.5, 30.5, 0, 18])
plt.legend(bbox_to_anchor=(0.06, 0.8), loc=2, borderaxespad=0,
           edgecolor='k', fancybox=False)
plt.show()