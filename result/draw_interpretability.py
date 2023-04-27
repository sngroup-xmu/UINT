# -*- coding: utf-8 -*-
import numpy as np
import statsmodels.api as sm # recommended import according to the docs
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn
if __name__ == '__main__':                          
      labels=['32*16', '32*32', '48*16', '48*32']
      df = pd.read_csv('data.csv')
      x3 = df['attack 0.5'].tolist()
      x4 = df['attack 0.9'].tolist()
      x5 = df['anchor 0.5'].tolist()
      x6 = df['anchor 0.9'].tolist()
      x1 = df['decision boundary 0.5'].tolist()
      x2 = df['decision boundary 0.9'].tolist()
      x7 = df['sensitivity 0.5'].tolist()
      x8 = df['sensitivity 0.9'].tolist()
      # x3 = df['Epsilon'][df["History"] == 5].tolist()
      # x4 = df['Epsilon'][df["History"] == 10].tolist()
      # x3 = df['Epsilon'][df["Features"] == "['send_ratio']"].tolist()
      # x4 = df['Epsilon'][df["Features"] == "['latency_inflation', 'latency_ratio', 'send_ratio']"].tolist()
      #=============绘制cdf图===============
      # x1=[i * 100 for i in x11]
      # x2 = [i * 100 for i in x22]
      # x1 = [i+1 for i in x1]
      # x2 = [i+1 for i in x2]
      x = np.arange(len(labels))  # the label locations
      width = 0.15
      # x3=[i/2 for i in x1]
      # x4=[i/2 for i in x2]
      fig, ax = plt.subplots()
      rects4 = plt.bar(x-width-width  , x4, width, label='Feature importance(90%)',color='#82B0D2')
      rects3 = plt.bar(x-width-width , x3, width, label='Feature importance(50%)',color='#FA7F6F')
      rects6 = plt.bar(x-(width)/2-width/4 , x6, width, label='Anchor(90%)',color='#EDE683')
      rects5 = plt.bar(x-(width)/2-width/4, x5, width, label='Anchor(50%)',color='#6EDBA7')
      rects2 = plt.bar(x+width/2 , x2, width, label='Decison boundary(90%)',color='#57D4CF')
      rects1 = plt.bar(x+width/2, x1, width, label='Decision boundary(50%)',color='#FFA47B')
      rects8 = plt.bar(x+width+3*(width)/4 , x8, width, label='Sensitivity analysis(90%)',color='#ff9bb1')
      rects7 = plt.bar(x+width+3*(width)/4, x7, width, label='Sensitivity analysis(50%)',color='#CFB0F5')
      # rects2 = plt.bar(x-width , x2, width, label='Decison boundary(90%)',color='#BC8379')
      # rects1 = plt.bar(x-width , x1, width, label='Decision boundary(50%)',color='#FA7F6F')
      # rects4 = plt.bar(x-width/2-width/4 , x4, width, label='Feature importance(90%)',color='#AD8A68')
      # rects3 = plt.bar(x-width/2-width/4, x3, width, label='Feature importance(50%)',color='#FFBE7A')
      # rects6 = plt.bar(x+(width) , x6, width, label='Anchor(90%)',color='#679C81')
      # rects5 = plt.bar(x+(width), x5, width, label='Anchor(50%)',color='#6EDBA7')
      plt.tick_params(labelsize=12)
      ax.set_ylabel('Time(s)',fontsize=12)
      ax.set_xlabel('Aurora Structure',fontsize=12)
      ax.set_xticks(x)
      ax.set_xticklabels(labels)
      plt.yscale('log')  # 设置纵坐标的缩放
      y = [pow(10, i) for i in range(1, 7)]
      # plt.ylim(0,100000)
      # ax.legend(ncol=2)
      plt.rcParams.update({'font.size': 12})
      # plt.legend(loc=9, bbox_to_anchor=(0.5,1.22),borderaxespad = 0,ncol=2)
      plt.legend(bbox_to_anchor=(0.5, -0.3),loc=10,ncol=2)
      # plt.ylim(0,1000000)
    #   ax.legend()
      plt.savefig('aurora_interpretability.pdf',dpi=600,bbox_inches='tight')
      plt.show()
