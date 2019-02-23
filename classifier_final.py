# random_forest_classifier.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import cross_validation
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import mlab
import itertools
import time
from pylab import *  
from matplotlib.ticker import MultipleLocator, FormatStrFormatter  
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore")

font = {'family' : 'serif',  
        'color'  : 'darkred',  
        'weight' : 'normal',  
        'size'   : 14,  
        }  
font_text = {'family' : 'serif',  
        'color'  : 'darkred',  
        'weight' : 'normal',  
        'size'   : 8,  
        }
class_name = ["action-1", "action-2", "action-3", "action-4", "action-5", "action-6", \
                "action-7", "action-8", "action-9", "action-10", "action-11", "action-12", \
                "action-13", "action-14", "action-15", "action-16"]

# class_name = ["action-1", "action-2", "step-3", "step-4", "step-5", "step-6", \
#                 "step-7", "step-8", "step-9", "step-10", "step-11", "step-12", \
#                 "step-13", "step-14", "step-15", "step-16"]

train = pd.read_csv("FS_bg_training.csv")
test = pd.read_csv("FS_bg_testing.csv")

# train = pd.read_csv("SCI_train_0304")
# test = pd.read_csv("SCI_test_0304")

train_data = np.array(train.iloc[:train.shape[0], :train.shape[1] - 1]) # 特征集
train_target = np.array(train.iloc[:train.shape[0], train.shape[1] - 1:])  # 标记
train_target = train_target.reshape(train_target.shape[0],)

test_data = np.array(test.iloc[:test.shape[0], :test.shape[1] - 1]) # 特征集
test_target = np.array(test.iloc[:test.shape[0], test.shape[1] - 1:])  # 标记
test_target = test_target.reshape(test_target.shape[0],)


classifier = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=6, random_state=1)
# start = time.clock()
classifier.fit(train_data, train_target)
# end = time.clock()
# print('train time: %s Seconds'%(end-start))
# start_y = time.clock()
y_predict = classifier.predict(test_data)
# end_y = time.clock()
# print('test time: %s Seconds'%(end_y-start_y))

sc = classifier.score(test_data, test_target)
print(sc)

# 混淆矩阵
# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     # print(cm)

#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, class_name, rotation=315, fontsize=10)
#     plt.yticks(tick_marks, class_name, fontsize=10) 

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         if cm[i, j] > 0:
#             plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  size=10,
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout(pad=1.5)
#     plt.ylabel('True label', fontdict=font)
#     plt.xlabel('Predicted label', fontdict=font)

# # Compute confusion matrix
# cnf_matrix = confusion_matrix(test_target, y_predict)
# np.set_printoptions(precision=2)

# # Plot non-normalized confusion matrix
# plt.figure(figsize=(10, 8), dpi=120)
# plot_confusion_matrix(cnf_matrix, classes=class_name)

# # Plot normalized confusion matrix
# plt.figure(figsize=(10, 8), dpi=120)
# plot_confusion_matrix(cnf_matrix, classes=class_name, normalize=True)
# plt.savefig("confusion_matrix_v5.jpg")


# 不同分类器 + 对比实验
# score_my = []
# score_sci = []
# classifiers = [
#     KNeighborsClassifier(5),
#     svm.SVC(kernel="linear", C=0.025),
#     RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=6, random_state=1),
#     GaussianNB()]
# for clf in classifiers:
# 	clf.fit(train_data, train_target)
# 	# probas = classifier.predict_proba(test_data)
# 	score = clf.score(test_data, test_target)
# 	score_my.append(score)
# print(score_my)
font = {'family' : 'serif',  
        'color'  : 'darkred',  
        'weight' : 'normal',  
        'size'   : 12,  
        }  
font_text = {'family' : 'serif',  
        'color'  : 'darkred',  
        'weight' : 'normal',  
        'size'   : 8,  
        }
fig, ax = plt.subplots()
# # SCI
Y1 = [0.52886836027713624, 0.46651270207852191, 0.51963048498845266, 0.44572748267898382]
# Yan
Y2 = [0.81381733021077285, 0.87353629976580793, 0.95892271662763464, 0.86885245901639341]
X = np.arange(len(Y1))
plt.bar(X, Y1, width = 0.35, facecolor = 'lightskyblue', label="Fu's method")
plt.bar(X + 0.36,Y2, width = 0.35, facecolor = 'yellowgreen', label="Our method")
# 水平柱状图plt.barh，属性中宽度width变成了高度height
# 打两组数据时用+
# facecolor柱状图里填充的颜色
# edgecolor是边框的颜色
# 想把一组数据打到下边，在数据前使用负号
# plt.bar(X, -Y2, width=width, facecolor='#ff9999', edgecolor='white')
# 给图加text
label = ["KNN", "SVM", "RF", "GNB"]
for x,y in zip(X,Y1):
    plt.text(x, y, '%.2f' % y, ha='center', va= 'bottom', fontdict=font_text)
for x,y in zip(X,Y2):
    plt.text(x+0.35, y, '%.2f' % y, ha='center', va= 'bottom', fontdict=font_text)

plt.legend(loc='upper left') # , bbox_to_anchor=(0.9,0.1)
plt.xticks(range(len(Y1)), ("KNN", "SVM", "RF", "GNB"))
plt.xlabel('Different algorithms', fontdict=font)
plt.ylabel('Accuracy', fontdict=font)
plt.ylim(0,+1.05)
ax.set_xticks(X+0.17)
plt.savefig("differe_ML_compare_v4.pdf")

# RF 参数
# scores = []
# area = np.pi * (15 * 0.4)**2
# for n_estimator in arange(25, 2000, 25):
# 	classifier = RandomForestClassifier(n_estimators=n_estimator, max_depth=None, min_samples_split=6, random_state=1)
# # param_test = {'n_estimators':range(0,800,25)}
# # clf = GridSearchCV(classifier, param_test, cv=5)
# 	classifier.fit(train_data, train_target)
# 	score = classifier.score(test_data, test_target)
# 	print(score)
# 	scores.append(score)
# X = np.arange(25, 2000, 25)
# plt.plot(X, scores, linewidth='2', color='b', marker='*')
# plt.savefig("test.jpg")

# 分类精度
# classifier = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=6, random_state=1)
# classifier.fit(train_data, train_target)
# y_predict = classifier.predict(test_data)
# print(classification_report(test_target, y_predict))


X = np.arange(1, 17, 1)
Y_acc = [0.97,0.98,0.96,0.96,0.96,1.00,1.00,1.00,1.00,1.00,0.98,0.93,0.98,0.96,1.00,1.00]
Y_recall = [1.00,1.00,1.00,1.00,1.00,0.96,0.96,0.98,0.98,0.98,0.98,0.96,0.96,0.96,0.94,0.98]
Y_F1 = [0.99,0.99,0.98,0.98,0.98,0.98,0.98,0.99,0.99,0.99,0.98,0.94,0.97,0.96,0.97,0.99]

# color = ['r', 'b', 'g', 'c', 'm', 'r', 'b', 'g', 'c', 'm', 'r', 'b', 'g', 'c', 'm', 'b']
color = ['#7F00FF','#0FFFFF','#FF00FF','#7FFF00','#00008B','#DC143C','#006400',
		'#6495ED','#0000FF','#8AEEE2','#A52A2A','#DEB887','#5F9EA0','#D2691E','#A9A9A9','#649F00',]

# plt.bar(X, height=Y_F1, align='center', color=color, yerr=0.00001)
# plt.tight_layout(pad=3)
# # plt.ylabel('F1-score', fontdict=font)
# # plt.xlabel('Different actions', fontdict=font)
# plt.xticks(X, class_name, rotation=315, fontsize=8)
# for x,y in zip(X, Y_F1):
#   plt.text(x, y, '%.2f' % y, ha='center', va= 'bottom', fontdict=font_text)
# plt.savefig("F1_v2.pdf")

# plt.bar(X, height=Y_recall, align='center', color=color, yerr=0.00001)
# plt.tight_layout(pad=3)
# # plt.ylabel('Recall', fontdict=font)
# # plt.xlabel('Different actions', fontdict=font)
# plt.xticks(X, class_name, rotation=315, fontsize=8)
# for x,y in zip(X, Y_recall):
#   plt.text(x, y, '%.2f' % y, ha='center', va= 'bottom', fontdict=font_text)
# plt.savefig("recall_v2.pdf")

# plt.bar(X, height=Y_acc, align='center', color=color, yerr=0.00001)
# plt.tight_layout(pad=3)
# # plt.ylabel('Accuracy', fontdict=font)
# # plt.xlabel('Different actions', fontdict=font)
# plt.xticks(X, class_name, rotation=315, fontsize=8)
# for x,y in zip(X, Y_acc):
#   plt.text(x, y, '%.2f' % y, ha='center', va= 'bottom', fontdict=font_text)
# plt.savefig("accuracy_v2.pdf")

# # pre:[0.96,0.97,0.84,0.93,0.95,1.00,1.00,1.00,0.98,1.00,0.98,0.91,1.00,0.96,0.87,0.98]
# # rec:[0.99,1.00,1.00,1.00,0.95,0.98,0.96,0.88,0.98,0.98,0.98,0.96,0.92,0.98,0.72,0.98]
# # F1:[0.98,0.98,0.91,0.96,0.95,0.99,0.98,0.94,0.98,0.99,0.98,0.94,0.96,0.97,0.79,0.98]

area = np.pi * (15 * 0.4)**2
# burst threshold
# scores = []
# for i in np.arange(0.05, 2, 0.025):
#     train_file = './burst_threshold/Feature_train_0306_80' + str(i) + '.csv'
#     test_file = './burst_threshold/Feature_test_0306_80_' + str(i) + '.csv'
#     train = pd.read_csv(train_file)
#     test = pd.read_csv(test_file)

#     train_data = np.array(train.iloc[:train.shape[0], :train.shape[1] - 1]) # 特征集
#     train_target = np.array(train.iloc[:train.shape[0], train.shape[1] - 1:])  # 标记
#     train_target = train_target.reshape(train_target.shape[0],)

#     test_data = np.array(test.iloc[:test.shape[0], :test.shape[1] - 1]) # 特征集
#     test_target = np.array(test.iloc[:test.shape[0], test.shape[1] - 1:])  # 标记
#     test_target = test_target.reshape(test_target.shape[0],)

#     classifier = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=6, random_state=1)

#     classifier.fit(train_data, train_target)
#     score = classifier.score(test_data, test_target)
#     print(score)
#     scores.append(score)
	
	# 选择最优参数
	# param_test = {'n_estimators':range(10,400,30), 'max_depth':[None, 1, 5 ,10], 'min_samples_split':[2, 4, 6]}
	# param_test = {'random_state': range(10), 'n_estimators':range(10,400,10),}
	# clf = GridSearchCV(classifier, param_test, cv=5)
	# clf.fit(X_train, y_train)
	# print(clf.best_params_)

	# 一次训练，一次测试
	# classifier.fit(X_train, y_train)
	# score = classifier.score(X_test, y_test)
	# print(classifier.feature_importances_) # 每个特征的重要程度
	# # print(classifier.predict(X_test))
	# print("-----------------------------")
	# print("one train: %s" % score)
	# print("-----------------------------")
	

	# 十折交叉验证（多标准，返回字典）
	# scoring = ['precision_macro', 'recall_macro', 'accuracy', 'f1_macro']
	# # cv = StratifiedKFold(n_splits=5, shuffle=False, random_state=0)
	# cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=None)
	# scores = cross_validate(classifier, data, target, scoring=scoring, 
											# cv=cv, return_train_score=False)

	# print("10-fold   fit-time: %0.2f (+/- %0.2f)" % (scores['fit_time'].mean(), scores['fit_time'].std() * 2))
	# print("10-fold score-time: %0.2f (+/- %0.2f)" % (scores['score_time'].mean(), scores['score_time'].std() * 2))
	# print("10-fold  precosion: %0.2f (+/- %0.2f)" % (scores['test_precision_macro'].mean(), scores['test_precision_macro'].std() * 2))
	# print("10-fold     Recall: %0.2f (+/- %0.2f)" % (scores['test_recall_macro'].mean(), scores['test_recall_macro'].std() * 2))
	# print("10-fold   accuracy: %0.2f (+/- %0.2f)" % (scores['test_accuracy'].mean(), scores['test_recall_macro'].std() * 2))
	# print("10-fold   f1_macro: %0.2f (+/- %0.2f)" % (scores['test_f1_macro'].mean(), scores['test_recall_macro'].std() * 2))
	# print("-----------------------------")
	# acc.append(scores['test_accuracy'].mean())

# scores = [0.763648928818,0.759917920657,0.704802259887,0.677252584934,0.748788368336,0.7910321489,0.793013100437,0.83426443203,0.857003891051,0.886773547094,0.864337101747,0.891640866873,0.893704850361,0.866597724922,0.903125,0.936422413793,0.948913043478,0.946623093682,0.955337690632,0.960698689956,0.960220994475,0.961283185841,0.922566371681,0.975555555556,0.957730812013,0.968715083799,0.961883408072,0.949438202247,0.941507311586,0.967194570136,0.975056689342,0.933106575964,0.902383654938,0.919226393629,0.953302961276,0.968109339408,0.933789954338,0.97149372862,0.97588978186,0.974566473988,0.933874709977,0.940766550523,0.932479627474,0.974358974359,0.972027972028,0.97668997669,0.967327887981,0.974299065421,0.978922716628,0.97062279671,0.978798586572,0.960992907801,0.972748815166,0.979809976247,0.964285714286,0.970131421744,0.970131421744,0.97005988024,0.925570228091,0.968674698795,0.97466827503,0.97466827503,0.965018094089,0.967391304348,0.945586457074,0.972121212121,0.92354368932,0.938031591738,0.956151035323,0.965853658537,0.923076923077,0.966952264382,0.954656862745,0.971779141104,0.965601965602,0.965601965602,0.920147420147,0.970479704797]

# fd = open('test.txt', 'w+')

# fd.write(str(scores))

# xmajorLocator   = MultipleLocator(250) 
# #将x主刻度标签设置为20的倍数  
# xmajorFormatter = FormatStrFormatter('%d') 
# #设置x轴标签文本的格式  
# xminorLocator   = MultipleLocator(50) 
# #将x轴次刻度标签设置为5的倍数  
  
# ymajorLocator   = MultipleLocator(0.01) 
# #将y轴主刻度标签设置为0.5的倍数  
# ymajorFormatter = FormatStrFormatter('%1.2f')
#  #设置y轴标签文本的格式  
# yminorLocator   = MultipleLocator(0.00125) 
# #将此y轴次刻度标签设置为0.1的倍数  
  
# t = np.arange(25, 2000, 25)
# # t = arange(0.0, 100.0, 1)  
# # s = sin(0.1*pi*t)*exp(-t*0.01)  
  
# ax = subplot(111) #注意:一般都在ax中设置,不再plot中设置  
# plot(t,scores,color='b', marker='*', markersize=10, linewidth=3)  
  
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
# ax.yaxis.grid(True, which='minor') #y坐标轴的网格使用次刻度
# xlabel('The number of trees', fontdict=font)
# ylabel('Accuracy', fontdict=font)
# tight_layout(pad=1)
# # savefig('Burst_threshold_v4.pdf')
# savefig('n_trees_1.pdf')