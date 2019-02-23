# -- conding:utf-8 --

# 长度分布

import dpkt
import socket
import sys
import time
import math
import csv
import pandas as pd
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

# 不同文件不同标记
def lable_class(num, file_class):
    len_distribution = []
    class_value = -1

    if file_class == 'send red packet':
        class_value = 1
    if file_class == 'rev red packet':
        class_value = 1
    if file_class == 'send transfer':
        class_value = 2
    if file_class == 'rev transfer':
        class_value = 2
    if file_class == 'txt':
        class_value = 3
    if file_class == 'pic':
        class_value = 4

    return class_value

# pcap文件的处理
def printPcap(pcap, file_class):
    time_s = []
    len_distribution = []
    # ts_threshold = 0
    for (ts, buf) in pcap:
        eth = dpkt.ethernet.Ethernet(buf)
        time_local = time.localtime(ts)

        ip = eth.data
        #print(ts, temp)
        if ip.data.__class__.__name__ =='TCP':
            tcp = ip.data
            # if len(tcp.data) > 0 :
            #     len_distribution.append(len(tcp.data))
            # if len(tcp.data) > 0 and len(tcp.data) <= 400 :
                # num = num + 1
            # if len(tcp.data) > 0:
            len_distribution.append(len(tcp.data))
            # time_s.append(ts)
    length.append(len_distribution)
            # time_s.append(time_local)
    # time_fin = []
    # for t in time_s[0:100]:
    #     time_fin.append(t-time_s[0])
    
    # ret = time_fin[0]
    # time_finally = []
    # th_time = 0
    # burst_time = 0
    # for t in time_fin:
    #     if t - ret > 1:
    #         th_time = time_finally[-1] + 0.2
    #         time_finally.append(th_time)
    #         burst_time = t
    #     else:
    #         time_finally.append(th_time + t - burst_time)
    #     ret = t
    # print(time_finally)
    # burst = []
    # sum = 0
    # for i in len_distribution:
    #     sum += i
    #     burst.append(sum)

    # plt.plot(time_finally[0:100], burst[0:100])
    # length.append(len_distribution)
    # print(len(len_distribution), num)
    # print(len_distribution)
    # print(ts, time.strftime("%Y-%m-%d %H:%M:%S",time_s[0]))
    # print(ts, time.strftime("%Y-%m-%d %H:%M:%S",time_s[-1]))
    # print(time_s[0],'\n', time_s[-1])
    # plt.xlabel('Total packet length (bytes)', fontdict = font)
    # plt.ylabel('Timestamp', fontdict = font)
    # plt.tight_layout(pad=1)
    
    # xmajorLocator   = MultipleLocator(10) 
    # #将x主刻度标签设置为20的倍数  
    # xmajorFormatter = FormatStrFormatter('%d') 
    # #设置x轴标签文本的格式  
    # xminorLocator   = MultipleLocator(2.5) 
    # #将x轴次刻度标签设置为5的倍数  
      
    # ymajorLocator   = MultipleLocator(2500) 
    # #将y轴主刻度标签设置为0.5的倍数  
    # ymajorFormatter = FormatStrFormatter('%d')
    #  #设置y轴标签文本的格式  
    # yminorLocator   = MultipleLocator(500) 
    #将此y轴次刻度标签设置为0.1的倍数  
      
    # t = np.arange(25, 2000, 25)
    # t = arange(0.0, 100.0, 1)  
    # s = sin(0.1*pi*t)*exp(-t*0.01)  
    # x = arange(0, 50, 1)
    # ax = subplot(111) #注意:一般都在ax中设置,不再plot中设置
    # th = burst[0]
    # for i in x:
    #     if th == burst[i]:
    #         plot(i, burst[i], color='b', linewidth=2)
    #     else:
    #         plot(i, burst[i], color='r', linewidth=2)
    #     th = burst[i]
      
    # #设置主刻度标签的位置,标签文本的格式  
    # ax.xaxis.set_major_locator(xmajorLocator)  
    # ax.xaxis.set_major_formatter(xmajorFormatter)  
      
    # ax.yaxis.set_major_locator(ymajorLocator)
    # ax.yaxis.set_major_formatter(ymajorFormatter)
      
    # #显示次刻度标签的位置,没有标签文本  
    # ax.xaxis.set_minor_locator(xminorLocator)
    # ax.yaxis.set_minor_locator(yminorLocator)

    # ax.grid(which='major', axis='x', linewidth=0.6, alpha=0.6, linestyle='--', color='#000000')
    # ax.grid(which='minor', axis='x', linewidth=0.4, alpha=0.8, linestyle='--', color='#000000')
    # ax.grid(which='major', axis='y', linewidth=0.6, alpha=0.6, linestyle='--', color='#000000')
    # ax.grid(which='minor', axis='y', linewidth=0.4, alpha=0.8, linestyle='--', color='#000000')
    # ax.xaxis.grid(True, which='major') #x坐标轴的网格使用主刻度  
    # ax.yaxis.grid(True, which='major') #y坐标轴的网格使用次刻度
    # ylabel('Total payload (bytes)', fontdict=font)
    # xlabel('Time series', fontdict=font)
    # # savefig('Burst_threshold_v4.pdf')
    # tight_layout(pad=1)
    # savefig('burst_v2.jpg')

print("FS start\n")

length = []
pic = open('./Data/Dataset_backup/data_third_step/pic_0114.pcap', 'rb')
txt = open('./Data/Dataset_bg/testing/txt_1122.pcap', 'rb')
red = open('./Data/Dataset_backup/data_third_step/red_0114.pcap', 'rb')
tran = open('./Data/Dataset_backup/data_third_step/tran_0114.pcap', 'rb')

printPcap(dpkt.pcap.Reader(txt), 'txt')
printPcap(dpkt.pcap.Reader(pic), 'pic')
printPcap(dpkt.pcap.Reader(red), 'red')
printPcap(dpkt.pcap.Reader(tran), 'tran')
# 
area = np.pi * (15 * 0.4)**2
# 长度分布
for i in range(200):
    if i != 199:
        if length[0][i] > 0:
            plt.scatter(i, length[0][i], marker = 'o', color = 'b')
        if length[1][i] > 0:
            plt.scatter(i, length[1][i], marker = '+', color = 'y')
        if length[2][i] > 0:
            plt.scatter(i, length[2][i], s= area, marker = '^', color = 'r')
        if length[3][i+100] > 0:
            plt.scatter(i, length[3][100+i], s= area, marker = '*', color = 'g')
    else:
        # if length[0][i] > 0:
            plt.scatter(i, length[0][i], marker = 'o', color = 'b', label='文本')
        # if length[1][i] > 0:
            plt.scatter(i, length[1][i], s= area, marker = '+', color = 'y', label="图片")
        # if length[2][i] > 0:
            plt.scatter(i, length[2][i], s= area, marker = '^', color = 'r', label="红包")
        # if length[3][i+100] > 0:
            plt.scatter(i, length[3][100+i], s= area, marker = '*', color = 'g', label="转账")


# plt.scatter(range(200), length[0][0:200], marker = 'o', label="texts")
# plt.scatter(range(200), length[1][0:200], marker = '+', label="pictures")
# plt.scatter(range(200), length[2][0:200], s= area, marker = '^', label="red packets")
# plt.scatter(range(200), length[3][100:300], s= area, marker = '*', label="fund transfers")

# plt.plot(range(200), length[0][0:200], label="txt")
# plt.plot(range(200), length[1][0:200], label="pic")
# plt.plot(range(200), length[2][0:200], label="red")
# plt.plot(range(200), length[3][100:300], label="trans")
plt.xlabel("时间序列", fontdict=font)
plt.ylabel("报文长度（字节）", fontdict=font)
plt.tight_layout(pad=0.5)
plt.legend(loc='lower right')

plt.savefig("len_distribution.jpg")
print("FS end\n")