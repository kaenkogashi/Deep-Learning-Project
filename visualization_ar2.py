import cv2
import numpy as np
import pandas as pd
import os
import csv
from statistics import mean, median,variance,stdev


# 元ビデオファイル読み込み
#画像ファイルとcsvを入れる
cap = cv2.VideoCapture("190226100257-190226100344.avi")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('video2.avi', fourcc, 28.0, (1280, 720))

count = 0
with open('190226100257-190226100344.csv') as f:
    reader = csv.reader(f)
    header = next(reader)
    #frame,x,y,w,h
    rawx = []
    rawy = []
    prex = []
    prey = []
    predict_flg = False
    predict_count = 27
       
    for row in reader:
        while(True):
            ret, img =cap.read()          
            if count == int(row[0]):#最初の行はskipする
                break
            if ret == False:
                break
            out.write(img)
            count+=1

        x = float(row[1])#+float(row[3])/2
        y = float(row[2])#+float(row[4])/2
        #raw配列に生値を入れる
        if len(rawx)<predict_count:
            rawx.append(x)
            rawy.append(y)
        elif len(rawx)==predict_count:
            rawx.pop(0)
            rawy.pop(0)
            rawx.append(x)
            rawy.append(y)
            predict_flg=True
        for i in range(len(rawx)):
            img = cv2.circle(img, (int(rawx[i]), int(rawy[i])), 5, (255, 0, 0), -1)        

        #予想はここから#############
        if predict_flg:
            predict_x = int(((x-mean(rawx[0:9]))+(x-mean(rawx[9:18]))*2+(x-mean(rawx[18:27]))*3)/3)+int(x)#適当な式
            predict_y = int(((y-mean(rawy[0:9]))+(y-mean(rawy[9:18]))*2+(y-mean(rawy[18:27]))*3)/3)+int(y)
            cv2.arrowedLine(img,(int(x), int(y)), (predict_x, predict_y), color=(0, 0, 255), thickness=2)
        #ここまで####################
                    
        count+=1        
        out.write(img)

out.release()
    
    

        
                

        
            

