# -- conding:utf-8 --

# 用于处理pcap文件， 提取所需特征
# 1. 标记： 读取csv中的序号，标记每个packet的类别
# 2. 切分： 以时间间隔阈值切分流量为多个burst，根据标记信息标记类别。
# 3. 特征： 每个burst内流的数量，数据包数量， tcp连接次数，ack数量，载荷序列的统计量(综述，均值，最值，方差，标准差)

import dpkt
import socket
import sys
import time
import math
import csv
import pandas as pd
import numpy as np

# tco包内协议转字符串
def flags_trans(flag):
    if flag == 16:
        return "ACK"
    if flag == 24:
        return "PSH + ACK"
    if flag == 17:
        return "FIN + ACK"
    if flag == 2:
        return "SYN"
    if flag == 18:
        return "SYN + ACK"

    return "else"

# 判断是否是握手协议
def is_hand(flag):
    return  flags_trans(flag) == "SYN" or flags_trans(flag) == "SYN + ACK" \
            or flags_trans(flag) == "FIN + ACK"

# 去除tcp载荷中为0的元素，返回最终的列表长度
def list_no_Zero(list):
    i = 0
    while i < len(list):
        if list[i] != 0:
            #list.pop(i)
            return 1
            #i -= 1
        i += 1
    return 0

# 标记csv 读至 list
def class_judge(file):
    col = []
    reader = csv.reader(file)
    for row in reader:
        for i in range(len(row)-1):
            col.append([])
            col[i].append(int(row[i]))
            col[i].append(int(row[i+1]))

    i = 0
    while i < len(col):
        if len(col[i]) == 0:
            col.pop(i)
            i -= 1
        i += 1
    print(col)
    return col

# 红包和转账分为接收和发送两类
def class_Coarse_Grained(class_value):
    if class_value >=0 and class_value <= 4:
        return 1
    if class_value >4 and class_value <=9:
        return 2

# 红包和转账步骤分为多类
def Class_Fine_Grained_1(class_value):
    # print(class_value)
    # train no bg
    # if class_value == 1 or class_value == 0:  # red_send
    #     return 1
    # if class_value == 2 :  # 不考虑第三种，第三种易混合
    #     return 2
    # if class_value == 4:
    #     return 3
    # if class_value == 5:
    #     return 4
    # return -100

    # test no bg
    # if class_value == 0:
    #     return 1
    # if class_value == 1:  # red_send
    #     return 2
    # if class_value == 2:  # 不考虑第三种，第三种易混合
    #     return 3
    # if class_value == 3:
    #     return 4
    # if class_value == 4:
    #     return 5

    # test bg
    if class_value >= 0:
        return class_value + 1   
    return -100

def Class_Fine_Grained_2(class_value):
    if class_value == 0:  # red_send
        return 1
    if class_value == 1:  # 不考虑第三种，第三种易混合
        return 1
    if class_value == 2:
        return 2
    if class_value == 3:
        return 2
    if class_value == 4:
        return 3
    if class_value == 5:
        return 4
 
    return -100

def class_red_send(num):
    for class_value in range(len(csv_red_send)):
        print(class_value)
        i = 0
        while i < len(csv_red_send[class_value]):
            if class_value != 0 and class_value != 6:
                temp = 1
            else:
                temp = 0
            if num >= csv_red_send[class_value][i]+temp and num <= csv_red_send[class_value][i+1]:
                # print(csv_red_send[class_value][i]+temp, csv_red_send[class_value][i + 1], num, class_value)
                return 10 + Class_Fine_Grained_2(class_value)
                #return 30 + class_Coarse_Grained(class_value)
            i = i + 2
    return -1

def class_red_rev(num):
    for class_value in range(len(csv_red_rev)):
        i = 0
        while i < len(csv_red_rev[class_value]):
            if class_value != 0 and class_value != 6:
                temp = 1
            else:
                temp = 0
            if num >= csv_red_rev[class_value][i]+temp and num <= csv_red_rev[class_value][i+1]:
                #print(csv_red[class_value][i]+temp, csv_red[class_value][i + 1], num, class_value)
                #return 3
                return 20 + Class_Fine_Grained_1(class_value)
                #return 30 + class_Coarse_Grained(class_value)
            i = i + 2
    return -1

def class_tran_send(num):
    for class_value in range(len(csv_tran_send)):
        i = 0
        while i < len(csv_tran_send[class_value]):
            if class_value != 0 and class_value != 6:
                temp = 1
            else:
                temp = 0
            if num >= csv_tran_send[class_value][i]+temp and num <= csv_tran_send[class_value][i+1]:
                #print(csv_trans[class_value][i]+temp, csv_trans[class_value][i + 1], num, class_value)
                #return 4
                return 30 + Class_Fine_Grained_1(class_value)
                #return 40 + class_Coarse_Grained(class_value)
            i = i + 2
    return -1

def class_tran_rev(num):
    for class_value in range(len(csv_tran_rev)):
        i = 0
        while i < len(csv_tran_rev[class_value]):
            if class_value != 0 and class_value != 6:
                temp = 1
            else:
                temp = 0
            if num >= csv_tran_rev[class_value][i]+temp and num <= csv_tran_rev[class_value][i+1]:
                #print(csv_trans[class_value][i]+temp, csv_trans[class_value][i + 1], num, class_value)
                #return 4
                return 40 + Class_Fine_Grained_1(class_value)
                #return 40 + class_Coarse_Grained(class_value)
            i = i + 2
    return -1

# 有效载荷的相关统计量， 去掉载荷是0的数据包
def pcap_math(packet):
    dl = pd.Series(packet)

    # sum_payload = sum(packet)
    # avg = sum_payload / len(packet)
    # div = 0
    # for i in packet:
    #     div += (i - avg) * (i - avg)
    # stand = div ** 0.5
    if math.isnan(dl.skew()):
        sk = 0
    else:
        sk = dl.skew()
    if math.isnan(dl.kurt()):
        kt = 0
    else:
        kt = dl.kurt()
    if math.isnan(dl.std()):
        std = 0
    else:
        std = dl.std()

    #  + ',' + str(dl.max()) + ',' + str(dl.min())
    return str(dl.sum()) + ',' + str(dl.mean()) \
             + ',' + str(std) + ',' + str(sk) \
            + ',' + str(kt)

# 长度分布
def pcap_len_distribution(packet):
    len_dis = [0, 0, 0, 0, 0] # [0-128, 129-256, 257-512, 513-1024, 1025-]
    for pac in packet:
        if pac <= 128:
            len_dis[0] += 1
        elif pac <= 256:
            len_dis[1] += 1
        elif pac <= 512:
            len_dis[2] += 1
        elif pac <= 1024:
            len_dis[3] += 1
        else:
            len_dis[4] += 1 
    return str(len_dis[0]) + ',' + str(len_dis[1]) + ',' + str(len_dis[2]) + ',' + \
            str(len_dis[3]) + ',' + str(len_dis[4])

def pcap_len_distribution_2(packet):
    bins, bins_edge = np.histogram(packet, bins=5)
    return str(bins[0]) + ',' + str(bins[1]) + ',' + str(bins[2]) + ',' + \
            str(bins[3]) + ',' + str(bins[4])

# 流的相关统计量
def flow_statics(flow):
    ret = []
    ret.append(len(flow))
    return ret

# 不同文件不同标记
def lable_class(num, file_class):
    class_value = -1

    if file_class == 'send red packet':
        class_value = class_red_send(num)
    if file_class == 'rev red packet':
        class_value = class_red_rev(num)
    if file_class == 'send transfer':
        class_value = class_tran_send(num)
    if file_class == 'rev transfer':
        class_value = class_tran_rev(num)
    if file_class == 'txt':
        class_value = 1
    if file_class == 'pic':
        class_value = 2

    return class_value

# pcap文件的处理
def printPcap(pcap, file_class):
    pop = []
    push = []
    class_value = -1
    num = 0
    temp = -2.0
    flow_map = {}
    time_map = {}
    flow_stat = []
    flow_info = []
    pop_burst = [0, 0]
    push_burst = [0, 0]
    global burst_order

    # 解析pcap文件
    for (ts, buf) in pcap:  # 时间戳和其他内容
        num += 1   # 数据包序号

        time_local = time.localtime(ts)

        eth = dpkt.ethernet.Ethernet(buf)  # 字节流 -> 以太网帧 
        ip = eth.data  # 以太网帧  -> IP packet

        if ip.data.__class__.__name__ =='TCP':  # tcp报文

            ip_src = socket.inet_ntoa(ip.src)  # ip转换
            ip_dst = socket.inet_ntoa(ip.dst)

            # 根据数据包序号对不同的流量类型进行标记
            class_value = lable_class(num, file_class)

            tcp = ip.data # IP -> TCP

            # 四元组，IP PORT
            ip_string = str(ip_src) + ':' + str(tcp.sport) + '--' + str(ip_dst) + ':' + str(tcp.dport)


            if ts-temp <= burst_threshold:   # packet time interval

                # 不同方向的报文总量，以及数量
                if ip_src == '192.168.23.2':
                    push_burst[0] += len(tcp.data)
                    push_burst[1] += 1
                if ip_dst == '192.168.23.2':
                    pop_burst[0] += len(tcp.data)
                    pop_burst[1] += 1

                temp = ts 
                # 切分burst, <key=burst_order, value=features>  
                if time_map.get(burst_order):
                    features = time_map.get(burst_order)
                    features[0].append(len(tcp.data))  # 载荷长度
                    features[1].append(ts)             # 时间戳信息
                    features[2].append(flags_trans(tcp.flags))   # tcp标志
                    if is_hand(tcp.flags):
                        features[3] = features[3] + 1              # tcp连接数量

                    # burst内的flow
                    if flow_map.get(ip_string):
                        flow_info = flow_map.get(ip_string)
                        flow_info[0] = flow_info[0] + 1    # 流内数据包数量
                        flow_info[1].append(len(tcp.data))   # 流内包载荷
                    else:
                        flow_map.setdefault(ip_string, [1, [len(tcp.data)]])

            else:
                burst_order = burst_order + 1  # 下一个burst
                temp = ts

                # 每个burst对应的 flow的处理
                if len(flow_map) != 0:
                    flow_stat.append(flow_statics(flow_map))  # flow的相关统计量
                    flow_map = {}      # 清空        

                # burst的第一个packet的处理
                if flow_map.get(ip_string):
                    flow_info = flow_map.get(ip_string)
                    flow_info[0] = flow_info[0] + 1
                    flow_info[1].append(len(tcp.data))
                else:
                    flow_map.setdefault(ip_string, [1, [len(tcp.data)]])

                if is_hand(tcp.flags):
                    time_map.setdefault(burst_order, [[len(tcp.data)], [ts], [flags_trans(tcp.flags)], 1, class_value])
                else:
                    time_map.setdefault(burst_order, [[len(tcp.data)], [ts], [flags_trans(tcp.flags)], 0, class_value])

                pop.append(pop_burst)
                push.append(push_burst)
                pop_burst = [0, 0]
                push_burst = [0, 0]
                
    flow_stat.append(flow_statics(flow_map))    # 最后一个flow

    txt_num = 0
    pic_num = 0
    red_num = 0
    trans_num = 0
    i = 0   # 处理流信息

    for key, value in time_map.items():
    
        packet_num = len(value[0])  # 数据包的数量

        if value[-1] <= -1:
            i += 1
            continue
        if len(value[0]) <= 3:
            i += 1
            # print(value[0], value[-1])
            continue
        if list_no_Zero(value[0]) == 0:   # 如果tcp载荷全部为0， 则剪枝，数据包数量仍然是真实的数据包数量
            i += 1
            # print(value[0], value[-1])
            continue

        ack_num = 0  # ack数量
        for ack in value[2]:
            if ack == "ACK":
                ack_num = ack_num + 1

        math_statics = pcap_math(value[0])      # 载荷的相关统计量
        len_distribution = pcap_len_distribution(value[0])      # 长度分布
        len_distribution_2 = pcap_len_distribution_2(value[0])
        time_duration = (value[1][-1] - value[1][0])    # 持续时间
        if time_duration != 0:
            avg_byte = sum(value[0])/time_duration      # 平均字节
        else:
            avg_byte = 0
        
        result.write(str(len_distribution) + ',' + str(len_distribution_2) + ',' + str(packet_num) + ',' + \
            str(value[3]) + ',' + str(ack_num) + ',' + str(math_statics) + ',' + str(time_duration) + ',' + \
            str(avg_byte) + ',' + str(pop[i][0]) + ',' + str(pop[i][1]) + ',' + str(push[i][0]) + ',' + \
            str(push[i][1]) + ',' + str(value[-1]) + '\n')
        i += 1



print("FS start\n")
burst_order = 0
burst_threshold = 1.25


# test  no background
# pic = open('./data_third_step/pic_0114.pcap', 'rb')
# txt = open('./data_second_step/txt_1122.pcap', 'rb')
# send_rp = open('./data_four_step/send_red_packet_0303.pcap', 'rb')
# rev_rp = open('./data_four_step/rev_red_packet_0303.pcap', 'rb')
# send_trans = open('./data_four_step/send_transfer_0303.pcap', 'rb')
# rev_trans = open('./data_four_step/rev_transfer_0303.pcap', 'rb')

# csv_red_send = open('./data_four_step/send_red_packet_0303.csv', 'r')
# csv_red_rev = open('./data_four_step/rev_red_packet_0303.csv', 'r')
# csv_tran_rev = open('./data_four_step/rev_transfer_0303.csv', 'r')
# csv_tran_send = open('./data_four_step/send_transfer_0303.csv', 'r')

# train no background
# pic = open('./data_third_step/pic_0114.pcap', 'rb')
# txt = open('./data_second_step/txt_1122.pcap', 'rb')
# send_rp = open('./data_third_step/red_0114_send.pcap', 'rb')
# rev_rp = open('./data_third_step/red_0114_rev.pcap', 'rb')
# send_trans = open('./data_third_step/tran_0114_send.pcap', 'rb')
# rev_trans = open('./data_third_step/tran_0114_rev.pcap', 'rb')

# csv_red_rev = open('./data_third_step/red_0114_rev.csv', 'r')
# csv_red_send = open('./data_third_step/red_0114_send.csv', 'r')
# csv_tran_rev = open('./data_third_step/tran_0114_rev.csv', 'r')
# csv_tran_send = open('./data_third_step/tran_0114_send.csv', 'r')



# test   background
# pic = open('./Dataset_bg/testing/pic_0114.pcap', 'rb')
# txt = open('./Dataset_bg/testing/txt_1122.pcap', 'rb')
# send_rp = open('./Dataset_bg/testing/send_red_packet_0302.pcap', 'rb')
# rev_rp = open('./Dataset_bg/testing/rev_red_packet_0302.pcap', 'rb')
# send_trans = open('./Dataset_bg/testing/send_transfer_0302.pcap', 'rb')
# rev_trans = open('./Dataset_bg/testing/rev_transfer_0302.pcap', 'rb')

# csv_red_send = open('./Dataset_bg/testing/send_red_packet_0302.csv', 'r')
# csv_red_rev = open('./Dataset_bg/testing/rev_red_packet_0302.csv', 'r')
# csv_tran_rev = open('./Dataset_bg/testing/rev_transfer_0302.csv', 'r')
# csv_tran_send = open('./Dataset_bg/testing/send_transfer_0302.csv', 'r')

# train  background
pic = open('./Data/Dataset_bg/training/pic_0114.pcap', 'rb')
txt = open('./Data/Dataset_bg/training/txt_0114.pcap', 'rb')
send_rp = open('./Data/Dataset_bg/training/red_0114_send.pcap', 'rb')
rev_rp = open('./Data/Dataset_bg/training/red_0114_rev.pcap', 'rb')
send_trans = open('./Data/Dataset_bg/training/tran_0114_send.pcap', 'rb')
rev_trans = open('./Data/Dataset_bg/training/tran_0114_rev.pcap', 'rb')

csv_red_rev = open('./Data/Dataset_bg/training/red_0114_rev.csv', 'r')
csv_red_send = open('./Data/Dataset_bg/training/red_0114_send.csv', 'r')
csv_tran_rev = open('./Data/Dataset_bg/training/tran_0114_rev.csv', 'r')
csv_tran_send = open('./Data/Dataset_bg/training/tran_0114_send.csv', 'r')

csv_red_rev = class_judge(csv_red_rev)
csv_red_send = class_judge(csv_red_send)
csv_tran_rev = class_judge(csv_tran_rev)
csv_tran_send = class_judge(csv_tran_send)


# result = open('FS_bg_testing.csv', 'w+')
result = open('FS_bg_training_0811.csv', 'w+')
result.write("bin1" + "," + "bin2" + "," + "bin3" + "," + "bin4" + "," + "bin5" + "," + \
    "bin6" + "," + "bin7" + "," + "bin8" + "," + "bin9" + "," + "bin10" + "," + \
    "pac_num" + "," + "tcp_num" + "," + "ack_num" + "," + "sum_byte" + "," + "mean_payload" + "," + \
    "stand_payload" + "," + "skew_payload" + "," + "kurt_payload" + "," + "time_duration" + "," + \
    "avg_byte" + "," + "pop_byte" + "," + "pop_pac_num" + "," + "push_byte" + "," + "push_pac_num" + "," + \
    "class_value" + "\n")

printPcap(dpkt.pcap.Reader(txt), 'txt')
printPcap(dpkt.pcap.Reader(pic), 'pic')
printPcap(dpkt.pcap.Reader(send_trans), 'send transfer')
printPcap(dpkt.pcap.Reader(rev_trans), 'rev transfer')
printPcap(dpkt.pcap.Reader(send_rp), 'send red packet')
printPcap(dpkt.pcap.Reader(rev_rp), 'rev red packet')


# 不同阈值的特征生成
# for i in np.arange(0.05, 2, 0.025):
#     # filename = './burst_threshold/Feature_train_0306_80' + str(i) + '.csv'
#     filename = './burst_threshold/Feature_test_0306_80_' + str(i) + '.csv'
#     result = open(filename, 'w+')
#     result.write("bin1" + "," + "bin2" + "," + "bin3" + "," + "bin4" + "," + "bin5" + "," + \
#     "bin6" + "," + "bin7" + "," + "bin8" + "," + "bin9" + "," + "bin10" + "," + \
#     "pac_num" + "," + "tcp_num" + "," + "ack_num" + "," + "sum_byte" + "," + "mean_payload" + "," + \
#     "stand_payload" + "," + "skew_payload" + "," + "kurt_payload" + "," + "time_duration" + "," + \
#     "avg_byte" + "," + "pop_byte" + "," + "pop_pac_num" + "," + "push_byte" + "," + "push_pac_num" + "," + \
#     "class_value" + "\n")
#     burst_threshold = i
#     # train
#     pic = open('./data_third_step/pic_0114.pcap', 'rb')
#     txt = open('./data_second_step/txt_1122.pcap', 'rb')
#     # send_rp = open('./data_third_step/red_0114_send.pcap', 'rb')
#     # rev_rp = open('./data_third_step/red_0114_rev.pcap', 'rb')
#     # send_trans = open('./data_third_step/tran_0114_send.pcap', 'rb')
#     # rev_trans = open('./data_third_step/tran_0114_rev.pcap', 'rb')

#     # csv_red_rev = open('./data_third_step/red_0114_rev.csv', 'r')
#     # csv_red_send = open('./data_third_step/red_0114_send.csv', 'r')
#     # csv_tran_rev = open('./data_third_step/tran_0114_rev.csv', 'r')
#     # csv_tran_send = open('./data_third_step/tran_0114_send.csv', 'r')

#     # test
#     # pic = open('./data_third_step/pic_0114.pcap', 'rb')
#     # txt = open('./data_second_step/txt_1122.pcap', 'rb')
#     send_rp = open('./data_four_step/send_red_packet_0303.pcap', 'rb')
#     rev_rp = open('./data_four_step/rev_red_packet_0303.pcap', 'rb')
#     send_trans = open('./data_four_step/send_transfer_0303.pcap', 'rb')
#     rev_trans = open('./data_four_step/rev_transfer_0303.pcap', 'rb')

#     csv_red_send = open('./data_four_step/send_red_packet_0303.csv', 'r')
#     csv_red_rev = open('./data_four_step/rev_red_packet_0303.csv', 'r')
#     csv_tran_rev = open('./data_four_step/rev_transfer_0303.csv', 'r')
#     csv_tran_send = open('./data_four_step/send_transfer_0303.csv', 'r')


#     csv_red_rev = class_judge(csv_red_rev)
#     csv_red_send = class_judge(csv_red_send)
#     csv_tran_rev = class_judge(csv_tran_rev)
#     csv_tran_send = class_judge(csv_tran_send)

#     printPcap(dpkt.pcap.Reader(txt), 'txt')
#     printPcap(dpkt.pcap.Reader(pic), 'pic')
#     printPcap(dpkt.pcap.Reader(send_trans), 'send transfer')
#     printPcap(dpkt.pcap.Reader(rev_trans), 'rev transfer')
#     printPcap(dpkt.pcap.Reader(send_rp), 'send red packet')
#     printPcap(dpkt.pcap.Reader(rev_rp), 'rev red packet')

print("FS end\n")
