# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 17:26:31 2022

@author: ZeyuanCheng
"""
# equal frequency
def equifreq(arr1, m):    
    a = len(arr1)
    n = int(a / m)
    for i in range(0, m):
        arr = []
        for j in range(i * n, (i + 1) * n):
            if j >= a:
                break
            arr = arr + [arr1[j]]
        print(arr)
  
# equal width
def equiwidth(arr1, m):
    a = len(arr1)
    w = int((max(arr1) - min(arr1)) / m)
    min1 = min(arr1)
    arr = []
    for i in range(0, m + 1):
        arr = arr + [min1 + w * i]
    arri=[]
      
    for i in range(0, m):
        temp = []
        for j in arr1:
            if j >= arr[i] and j <= arr[i+1]:
                temp += [j]
        arri += [temp]
    print(arri) 
  
# data to be binned
#data = [5, 20, 5, 101, 9, 15, 42, 99, 7]
data = [5, 5, 7, 9, 15, 20, 42, 99, 101]

  
# no of bins
m = 3 
  
print("equal frequency binning")
equifreq(data, m)
  
print("\n\nequal width binning")
equiwidth(data, 3)