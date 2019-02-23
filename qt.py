# import pandas as pd
# import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib import mlab
# import itertools
# import time
# from pylab import *  
# from matplotlib.ticker import MultipleLocator, FormatStrFormatter  
# import warnings
# import sklearn.exceptions
# warnings.filterwarnings("ignore")

# font = {'family' : 'serif',  
#         'color'  : 'black',  
#         'weight' : 'bold',  
#         'size'   : 12,  
#         }  
# font_text = {'family' : 'serif',  
#         'color'  : 'red',  
#         'weight' : 'bold',  
#         'size'   : 10,  
#         }


# X = ["Nikon D70", "Nikon D70s", "Nikon D200", "Canon 100D", "Pentax K-50", "Nikon D5200"]
# Y = [0.85, 0.91, 1.56, 4.03, 2.928, 5.51]
# color = ['#006400','#0FFFFF','#FF00FF','#7FFF00','#00008B','#DC143C']
# X_1 = np.arange(6)
# plt.bar(X_1, Y, align='center', color=color, yerr=0.00001, tick_label=X, width=0.52)
# plt.tight_layout(pad=3)
# plt.ylabel('Time(s)', fontdict=font)
# plt.xticks(fontsize=8, fontweight='bold', color='black')
# plt.yticks(fontsize=8, fontweight='bold', color='black')
# for x,y in zip(X_1, Y):
#   print(x, y)
#   plt.text(x, y, '%.2f' % y, ha='center', va= 'bottom', fontdict=font_text)
# plt.savefig("qt.pdf")

import matplotlib
str=matplotlib.matplotlib_fname()
print(str)