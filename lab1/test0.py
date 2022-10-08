# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 15:03:52 2022

@author: ZeyuanCheng
"""
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("E:/ANU/Y2_2022_s2/COMP8430 Data Wrangling/Lab/lab1/weather_sample.csv")

print(df.head())

print(df[10:20])

#df['column1'].plot(kind='hist', bins=50) or df['column1'].plot.hist(bins=50)