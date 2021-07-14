import matplotlib.pyplot as plt


# 画图 #########################
x = [16,18,20,22,24,26,28,30]
y11 = [ 445, 844,1263, 707, 578,1212, 868, 465]
y21 = [7117,7937,7444,7264,6353,7186,6290,5987]
y31 = [3181,2942,3037,2849,2554,3707,2797,2420]
y12 = [2605,1848,1469, 707, 578,1212, 868, 465]
y22 = [8327,7548,7527,7264,6353,7186,6290,5987]
y32 = [3590,3548,2892,2849,2554,3707,2797,2420]

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

plt.title("Comparison of moving cost")
plt.xlabel("Maximum number of nodes in each block")
plt.ylabel("Moving cost")
plt.axis([14, 32, 0, 9000])
plt.grid(b=True, axis='y')  # 只显示y轴网格线
plt.legend(loc="upper right")  # lower   right
plt.show()
