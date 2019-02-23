实验文件夹;

# compare experiment:
0111_sci_data: 	SCI生成的一些数据。
labels: 		SCI实验的标记数据。 
code: 			SCI experiment code

# my experiment

Dataset_no_bg:  	未消除背景流量的实验数据集以及标记信息。
Dataset_bg:			消除背景流量的实验数据集以及标记信息。
Dataset_backup:		所以数据集的备份。
labels:				特征向量。
burst_threshold: 	阈值影响。

# code

- classifier_final.py
根据最终生成的特征向量建立分类器，计算混淆矩阵，阈值影响，对比实验，RF参数影响， 最终分类精度。

- FS_handler_final.py
处理流量。按照burst threshold切分流量。并且计算每一个burst内的特征值形成特征向量。

- pcap_label.py
生成数据的标记信息，仍需手动修改。不够准确。

- len_distribution.py
计算长度分布。

