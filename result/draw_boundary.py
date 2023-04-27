# -*- coding: utf-8 -*-
import numpy as np
import statsmodels.api as sm # recommended import according to the docs
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn
import numpy as np
import matplotlib.pyplot as plt 
from scipy.ndimage import gaussian_filter1d
if __name__ == '__main__':                          
    #   labels=['32*16', '32*32', '48*16', '48*32']
      df = pd.read_csv('decision_boundaries_aurora0_1.csv')
      # x1 = df['decision boundary 0.5'].tolist()
      # x2 = df['decision boundary 0.9'].tolist()
      x = df['latency_inflation_1'].tolist()
      y = df['latency_ratio_1'].tolist()
      y_smoothed = gaussian_filter1d(y, sigma=5)
      plt.xlim(-0.65,0.29)
      plt.ylim(8.5,9.3)  
      plt.fill_between(x, y_smoothed, 9.3, color='#FFACBE', alpha=0.5)
      plt.fill_between(x, y_smoothed, 0, color='#A4C4FF', alpha=0.8)
    #   plt.plot(x, y_smoothed)
    #   plt.title("Spline Curve Using the Gaussian Smoothing")
      plt.rcParams.update({'font.size': 12})  
      plt.xlabel("Latency Inflation",size=12)
      plt.ylabel("Latency Ratio", size=12)
      plt.savefig('aurora_boundary.pdf',dpi=600,bbox_inches='tight')
      plt.show()