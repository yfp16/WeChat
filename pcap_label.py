# 用于标记流量 

import dpkt
import csv
import time
import math
import socket


def preprocess_test(data):
	num = 0
	burst = []
	burst_num = []
	time_temp = 0
	flag_temp = ''
	seq_temp = 0  # tcp重传
	ack_temp = 0
	len_temp = 0
	for (ts, buf) in data:
		num += 1
		time_local = time.localtime(ts)
		eth = dpkt.ethernet.Ethernet(buf)
		ip = eth.data
		if ip.data.__class__.__name__ == 'TCP':
			tcp = ip.data
			print(time_local, ts)
			# if seq_temp == tcp.seq and ack_temp == tcp.ack and flag_temp == tcp.flags and len_temp == len(tcp.data):
			# 	print(num, tcp.seq, tcp.ack)
			# elif ts - time_temp < 1:
			# 	time_temp = ts
			# 	burst_num.append(num)
			# 	seq_temp = tcp.seq
			# 	ack_temp = tcp.ack
			# 	flag_temp = tcp.flags
			# 	len_temp = len(tcp.data)
			# else:
			# 	time_temp = ts
			# 	if len(burst_num) > 0:
			# 		burst.append(burst_num)
			# 	burst_num = []
			# 	burst_num.append(num)
			# 	seq_temp = tcp.seq
			# 	ack_temp = tcp.ack
			# 	flag_temp = tcp.flags
			# 	len_temp = len(tcp.data)

def preprocess(data):
	num = 0
	time_temp = 0
	burst = []
	burst_num = []
	for (ts,buf) in data:
		num += 1
		time_local = time.localtime(ts)
		eth = dpkt.ethernet.Ethernet(buf)
		ip = eth.data

		if ip.data.__class__.__name__ == 'TCP':
			if ts - time_temp < 1:
				burst_num.append(num)
				time_temp = ts
			else:
				time_temp = ts
				if len(burst_num) > 2:
					burst.append(burst_num)
				burst_num = []
				burst_num.append(num)

	# print(burst)
	b = [1]
	for bur in burst:
		print(bur[0], bur[-1])
		b.append(bur[-1])
	print(b)
	for i in b:
		# w = b[i*6 : (i+1)*6]
		# for i in range(len(w)):
		# 	red.write(str(w[i]))
		# 	if i == len(w):
		# 		red.write('\n')
		# 	else:
		# 		red.write(',')
		file_w.write(str(i)+',')

print('------start------')

file = open('./data_four_step/send_red_packet_0303.pcap', 'rb')
file_w = open('./data_four_step/send_red_packet_0303.csv', 'w+')
preprocess(dpkt.pcap.Reader(file))
print('-------end-------')