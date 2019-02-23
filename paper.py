# -*- coding: utf-8 -*-
import numpy as np  

from matplotlib import pyplot as plt
from matplotlib import mlab
from matplotlib.ticker import MultipleLocator, FormatStrFormatter  
from pylab import *  
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family']='sans-serif'

font = {'family' : 'sans-serif',  
        'color'  : 'darkred',  
        'weight' : 'normal',  
        'size'   : 12,  
        }  
font_text = {'family' : 'sans-serif',  
        'color'  : 'darkred',  
        'weight' : 'normal',  
        'size'   : 10,  
        }
fig, ax = plt.subplots()

Y = [0.96, 0.97, 0.80, 0.82, 0.81, 0.83]
X = np.arange(len(Y))
plt.bar(X, Y, width = 0.5, facecolor = 'lightskyblue')
# plt.bar(X + 0.36,Y2, width = 0.35, facecolor = 'yellowgreen', label="Our method")
# 水平柱状图plt.barh，属性中宽度width变成了高度height
# 打两组数据时用+
# facecolor柱状图里填充的颜色
# edgecolor是边框的颜色
# 想把一组数据打到下边，在数据前使用负号
# plt.bar(X, -Y2, width=width, facecolor='#ff9999', edgecolor='white')
# 给图加text
label = ["文本", "图片", "发送红包", "接收红包", "发送转账", "接收转账"]
for x,y in zip(X,Y):
    plt.text(x, y+0.01, '%.2f' % y, ha='center', va= 'bottom', fontdict=font_text)
# for x,y in zip(X,Y2):
#     plt.text(x+0.35, y, '%.2f' % y, ha='center', va= 'bottom', fontdict=font_text)

plt.legend(loc='upper left') # , bbox_to_anchor=(0.9,0.1)
plt.xticks(range(len(Y)), label)
plt.ylim(0,+1.05)
plt.xlabel('微信用户行为', fontdict=font)
plt.ylabel('准确率', fontdict=font)
plt.savefig("dialog_acc.jpg")
# plt.ylabel('召回率', fontdict=font)
# plt.savefig("dialog_rec.jpg")
# plt.ylabel('F1值', fontdict=font)
# plt.savefig("dialog_f1.jpg")