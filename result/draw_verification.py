# -*- coding: utf-8 -*-
import numpy as np
import statsmodels.api as sm # recommended import according to the docs
import matplotlib.pyplot as plt
import pandas as pd
import math

if __name__ == '__main__':                          
      labels=['32*16', '32*32', '48*16', '48*32']
      df = pd.read_csv('data2.csv')
      x1 = df['adversarial 0.5'].tolist()
      x2 = df['adversarial 0.9'].tolist()
      x3 = df['mf 0.5'].tolist()
      x4 = df['mf 0.9'].tolist()
      # x5 = df['live 0.5'].tolist()
      # x6 = df['live 0.9'].tolist()
      x7 = df['mf2 0.5'].tolist()
      x8 = df['mf2 0.9'].tolist()
      # x3 = df['Epsilon'][df["History"] == 5].tolist()
      # x4 = df['Epsilon'][df["History"] == 10].tolist()
      # x3 = df['Epsilon'][df["Features"] == "['send_ratio']"].tolist()
      # x4 = df['Epsilon'][df["Features"] == "['latency_inflation', 'latency_ratio', 'send_ratio']"].tolist()
      #=============绘制cdf图===============
      # x1=[i * 100 for i in x11]
      # x2 = [i * 100 for i in x22]
    #   x1 = [i+1 for i in x1]
    #   x2 = [i+1 for i in x2]
      x = np.arange(len(labels))  # the label locations
      width = 0.2
      # x3=[i/2 for i in x1]
      # x4=[i/2 for i in x2]
      fig,ax = plt.subplots()
      rects2 = plt.bar(x-3*(width/2) , x2, width, label='Adversarial Perturbation(90%)', color='#82B0D2')
      rects1 = plt.bar(x-3*(width/2) , x1, width, label='Adversarial Perturbation(50%)', color='#FA7F6F')
      rects4 = plt.bar(x-width/4 , x4, width, label='Missing feature(90%)', color='#EDE683')
      rects3 = plt.bar(x-width/4, x3, width, label='Missing feature(50%)', color='#6EDBA7')
      # rects6 = plt.bar(x+(width/2) , x6, width, label='Liveness Property(90%)', color='#FFE5DC')
      # rects5 = plt.bar(x+(width/2), x5, width, label='Liveness Property(50%)', color='#FA7F6F')
      rects8 = plt.bar(x+(width) , x8, width, label='Missing feature(|K|=2)(90%)', color='#57D4CF')
      rects7 = plt.bar(x+(width), x7, width, label='Missing feature(|K|=2)(50%)', color='#FFA47B')
      # Add some text for labels, title and custom x-axis tick labels, etc.
      plt.tick_params(labelsize=12)
      ax.set_ylabel('Time(s)',fontsize=12)
      ax.set_xlabel('Aurora Structure',fontsize=12)
      ax.set_xticks(x)
      ax.set_xticklabels(labels)
      plt.yscale('log')  # 设置纵坐标的缩放
      y = [pow(10, i) for i in range(1, 7)]
      # plt.rcParams.update({'font.size': 12})
      # plt.legend(ncol=2)
      plt.rcParams.update({'font.size': 12})      
      # plt.legend(bbox_to_anchor=(1.35, 0.2),loc=10)
      plt.legend(bbox_to_anchor=(0.5, -0.3),loc=10,ncol=2)
      # plt.ylim(0,1000000)
    #   ax.legend()
      plt.savefig('aurora_verification.pdf',dpi=600,bbox_inches='tight')

      plt.show()
      
