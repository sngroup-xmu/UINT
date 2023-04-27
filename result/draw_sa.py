import numpy as np
import statsmodels.api as sm # recommended import according to the docs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import math

if __name__ == '__main__':                          
      labels=['latency \ninflation\n', 'latency\n ratio', 'send\n ratio']
      labels2=['previous\nbit rate','   current \n buffer \n size','throu\n-ghput','download \ntime','next \nchunk \nsize','remain\n chunks']
      df = pd.read_csv('sensitivity_aurora0_origin.csv')
      x1 = df['N(x)-y'].tolist()
      x1=[i / 100 for i in x1]
      df = pd.read_csv('sensitivity_pensieve_origin.csv')
      x2 = df['N(x)-y'].tolist()
      x2=[i / 100 for i in x2]
      x = np.arange(len(labels))
      y = np.arange(len(labels2))
      width = 0.3
      # fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
      # rects = ax1.bar(x, x1, width=0.2, color='#9EC3DB')
      # ax1.set_ylabel('The deviation of output',fontsize=12)
      # ax1.set_xlabel('(a)Aurora',fontsize=12)
      # ax1.set_xticks(x)
      # ax1.set_xticklabels(labels)
      # rects = ax2.bar(y, x2, width=0.4, color='#FB9489')
      # ax2.set_ylabel('The deviation of output',fontsize=12)
      # ax2.set_xlabel('(b)Pensieve',fontsize=12)
      # ax2.set_xticks(y)
      # ax2.set_xticklabels(labels2)
      # ax1.tick_params(labelsize=11)
      # plt.rcParams.update({'font.size': 12})
      # ax2.tick_params(labelsize=11)
      # # ax2.rcParams.update({'font.size': 12})
      # plt.tight_layout()
      # plt.savefig('sensitivity_analysis.pdf',dpi=600,bbox_inches='tight')
      # plt.show()
      fig = plt.figure(figsize=(10,5))  # 创建画布
      grid = gridspec.GridSpec(1, 3)  # 设定2行*3列的网格

      ax1 = fig.add_subplot(grid[0,0])  # 第一行的全部列都添加到ax1中
        # 在ax1中绘图与操作，这都是这个ax的操作，不会影响全局

      ax2 = fig.add_subplot(grid[0, 1:])  # 第二行，第1列
      
      rects = ax1.bar(x, x1, width=0.4, color='#9EC3DB')
      ax1.set_ylabel('The deviation of output',fontsize=16)
      ax1.set_xlabel('(a)Aurora',fontsize=16)
      ax1.set_xticks(x)
      ax1.set_xticklabels(labels)
      rects = ax2.bar(y, x2, width=0.4, color='#FB9489')
      ax2.set_ylabel('The deviation of output',fontsize=16)
      ax2.set_xlabel('(b)Pensieve',fontsize=16)
      ax2.set_xticks(y)
      ax2.set_xticklabels(labels2)
      ax1.tick_params(labelsize=16)
      plt.rcParams.update({'font.size': 16})
      ax2.tick_params(labelsize=16)
      # ax2.rcParams.update({'font.size': 12})
      ax1.plot()
      ax2.plot()
      plt.tight_layout()
      plt.savefig('sensitivity_analysis.pdf',dpi=600,bbox_inches='tight')
      plt.show()