import cv2
import numpy as np
import pandas as pd
import os

#ver人4分割
#人と物体のrectの重なり割合を算出する関数
def overlap(df,rect,name):
    for j in range(4):
        left="left"+str(j)
        right="right"+str(j)
        top="top"+str(j)
        bottom="bottom"+str(j)
        over_L="over_L"+str(j)
        over_R="over_R"+str(j)
        over_T="over_T"+str(j)
        over_B="over_B"+str(j)
        name2 = name + "-" + str(j)
        df[over_L]=df[left]
        df.loc[df[over_L]<rect[0], over_L] = rect[0]
        df[over_R]=df[right]
        df.loc[df[over_R]>rect[1], over_R] = rect[1]
        df[over_T]=df[top]
        df.loc[df[over_T]<rect[2], over_T] = rect[2]
        df[over_B]=df[bottom]
        df.loc[df[over_B]>rect[3], over_B] = rect[3]    
        df[name2]=(df[over_R]-df[over_L])*(df[over_B]-df[over_T])/((rect[1]-rect[0])*(rect[3]-rect[2]))
        df.loc[df[over_L]>df[over_R], name2] = 0
        df.loc[df[over_T]>df[over_B], name2] = 0
    return df
        
if __name__ == '__main__':

    ###########
    #初期設定
    ###########
    input_folder = "./dataset/kusakaGomiToCSV/cleaned/"
    output_folder ="./dataset/kusakaGomiToCSV/all/"

    #Objectのrectを指定
    rect=[]
    rect.append([0,200,0,500])#left,right,top,bottom
    rect.append([0,200,0,500])
    rect.append([0,200,0,500])
    rect.append([0,200,0,500])
    #rect.append([0,0,10,10]) #rect1必要であれば追加(以下rect2…を追加)

    #画像のサイズ
    w = 1280 
    h = 720

    ###########
    #ここまで
    ###########
    #dataを取得
    #dataを取得
    file_name = os.listdir(input_folder)
    print(file_name)
    for j in range(len(file_name)):
    #for j in range(1):
        df = pd.read_csv(input_folder+file_name[j],usecols=['x','y'])
        df["x"]= df["x"]/w*2-1
        df["y"]= df["y"]/h*2-1
        df.to_csv(output_folder+"all.csv", mode='a',header=False)
        '''
        #人のbouding boxを決める
        box_size = 10#１辺１０pixelの正方形、自分で決められる、注意：人のバウンディングボックスのサイズではない
        box_size =box_size/2 #グリッドの長さ。４つのグリッドの長さではない

        #左上の人rect
        df["left0"] =df["x"] - box_size
        df["right0"]=df["x"]
        df["top0"] = df["y"] - box_size
        df["bottom0"] =df["y"]
        #右上の人のrect
        df["left1"] =df["x"]
        df["right1"]=df["x"] + box_size
        df["top1"] = df["y"] - box_size
        df["bottom1"] =df["y"]
        #左下の人のrect
        df["left2"] =df["x"] - box_size
        df["right2"]=df["x"]
        df["top2"] = df["y"] - box_size
        df["bottom2"] =df["y"]
        #右下の人のrect
        df["left3"] =df["x"]
        df["right3"]=df["x"] + box_size
        df["top3"] = df["y"]
        df["bottom3"] =df["y"]+  box_size

        #rectと人のオーバーラップを決める
        for i in range(len(rect)):
            name ="overlap"+str(i)
            df=overlap(df,rect[i],name)

        #人の座標の正規化
        #-1-1の場合
        df["xn"]= df["x"]/w*2-1
        df["yn"]= df["y"]/h*2-1

        df2 = df.loc[:,["count","xn","yn","overlap0-0","overlap0-1","overlap0-2","overlap0-3",
                                        "overlap1-0","overlap1-1","overlap1-2","overlap1-3",
                                        "overlap2-0","overlap2-1","overlap2-2","overlap2-3",
                                        "overlap3-0","overlap3-1","overlap3-2","overlap3-3",]]
        #boundingboxが3以上の時はdf2"overlap3-0～を追加する
        df2.to_csv(output_folder+"n_"+file_name[j])
        '''