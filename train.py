import csv
import os
import glob
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import SGD
import math
import numpy as np
from sklearn.metrics import mean_squared_error

def mkDataSet(data_size, data_length=60):
    """
    params\n
    data_size : データセットサイズ 10000\n
    data_length : 各データの時系列長\n
    returns\n
    train_x : トレーニングデータ（t=1,2,...,size-1の値)\n
    train_t : トレーニングデータのラベル（t=sizeの値）\n
    """
    train_x = []
    train_t = []

    #data = pd.read_csv("C:\\Users\\010170243\\work\\seq2seq\\dataset\\kusakaGomiToCSV\\csv\\kusaka1.csv", usecols=['x','y'])
    data = pd.read_csv("C:\\Users\\010170243\\work\\seq2seq\\dataset\\kusakaGomiToCSV\\all\\all.csv", usecols=[1, 2])#usecols=['x','y'])
    # type *.csv > C:\Users\010170243\work\seq2seq\dataset\kusakaGomiToCSV\cleaned\all\kusaka_all.csv
    data_np = np.asarray(data)#元はasarray

    q = data_np.shape[0]//data_length 
    amari =data_np.shape[0]%data_length 
    print(q)
    print(amari)
    print(type(data_np[0][0]))

    train_t=data_np[data_length:data_np.shape[0]-amari:,:]#X[start:end:step]出力のこと
    train_x=data_np[0:data_length*(q-1):,:]#入力
    print(train_x.shape[0]%data_length )
    print(train_x.shape[0]//data_length )
    print(train_t.shape[0]%data_length )#x=x学習データ,t=y正解
    print(train_t.shape[0]//data_length )

    print(data_np.shape[0])
    #print(train_x)
    #print(data_np)
    #print(data_np.shape)
    #print(type(data_np))
    #data_np = np.resize(data_np, (q,60,2))#https://deepage.net/features/numpy-reshape.html
    #print(data_np)
    #print(data_np.shape)
    #print(type(data_np))

    train_t= np.resize(train_t, (q-1,data_length,2))
    train_x= np.resize(train_x, (q-1,data_length,2))
    #print(train_x)
    return train_x, train_t

def mkTestSet(data_size, data_length=60):
    """
    params\n
    data_size : データセットサイズ 10000\n
    data_length : 各データの時系列長\n
    returns\n
    train_x : トレーニングデータ（t=1,2,...,size-1の値)\n
    train_t : トレーニングデータのラベル（t=sizeの値）\n
    """
    test_x = []
    test_t = []

    #data = pd.read_csv("C:\\Users\\010170243\\work\\seq2seq\\dataset\\kusakaGomiToCSV\\csv\\kusaka1.csv", usecols=['x','y'])
    data = pd.read_csv("C:\\Users\\010170243\\work\\seq2seq\\dataset\\kusakaGomiToCSV\\all\\kusaka_test.csv", usecols=[1, 2])#usecols=['x','y'])
    # type *.csv > C:\Users\010170243\work\seq2seq\dataset\kusakaGomiToCSV\cleaned\all\kusaka_all.csv
    data_np = np.asarray(data)#元はasarray

    q = data_np.shape[0]//data_length 
    amari =data_np.shape[0]%data_length 
    print(q)
    print(amari)
    print(type(data_np[0][0]))

    test_t=data_np[data_length:data_np.shape[0]-amari:,:]#X[start:end:step]出力のこと
    test_x=data_np[0:data_length*(q-1):,:]#入力
    print(test_x.shape[0]%data_length )
    print(test_x.shape[0]//data_length )
    print(test_t.shape[0]%data_length )#x=x学習データ,t=y正解
    print(test_t.shape[0]//data_length )
    print(data_np.shape[0])

    test_t= np.resize(test_t, (q-1,data_length,2))
    test_x= np.resize(test_x, (q-1,data_length,2))
    return test_x, test_t

class Predictor(nn.Module):#モデル
    def __init__(self, inputDim, hiddenDim, outputDim):
        super(Predictor, self).__init__()

        self.rnn = nn.LSTM(input_size = inputDim,
                            hidden_size = hiddenDim,
                            batch_first = True)
        self.output_layer = nn.Linear(hiddenDim, outputDim)
    
    def forward(self, inputs, hidden0=None):#計算を伝搬して backwordは不要
        
        #inputs=inputs.type(torch.FloatTensor)#型を再度変換しないと通らない。
        output, hidden = self.rnn(inputs, hidden0)#出力は層を積まないかつ単方向
        #print(inputs)
        #print("====================================================")
        #output = self.output_layer(output[:, -1, :])#outputの一番最後の値だけなので、output[:, -1, :]で時系列の最後の値
        output = self.output_layer(output[:, :, :])# 全部取り出しをするX = np.arange(125).reshape(5,5,5)
        #print(output)
        return output

def mkRandomBatch(train_x, train_t, batch_size=1):#10->1
    """
    train_x, train_tを受け取ってbatch_x, batch_tを返す。
    """
    batch_x = []
    batch_t = []
    
    for _ in range(batch_size):
        idx = np.random.randint(0, len(train_x) - 1)
        batch_x.append(train_x[idx])
        batch_t.append(train_t[idx])
    #print("=================================")
    batch_x=np.asarray(batch_x)
    batch_t=np.asarray(batch_t)
    #batch_x=torch.from_numpy(batch_x)
    #batch_t=torch.from_numpy(batch_t)
    #print(train_t)
    #print(batch_t.shape)
    return torch.FloatTensor(batch_x), torch.FloatTensor(batch_t)

def main():
    training_size = 1#10000
    valid_size = 1#1000
    test_size = 15#1000
    epochs_num = 1#1000
    hidden_size = 60#5
    batch_size = 1#100
    data_length = 60

    train_x, train_t = mkDataSet(training_size)
    valid_x, valid_t = mkDataSet(valid_size)
    #print(valid_t)

    model = Predictor(2, hidden_size, 2)
    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs_num):
        # training
        running_loss = 0.0
        training_accuracy = 0.0
        for i in range(int(training_size / batch_size)):
            optimizer.zero_grad()# 勾配の初期化

            data, label = mkRandomBatch(train_x, train_t, batch_size)

            output = model(data)# 順伝播

            loss = criterion(output, label)# ロスの計算
            loss.backward()# 勾配の計算
            optimizer.step()# パラメータの更新

            running_loss += loss.item()
            #training_accuracy += np.sum(np.abs((output.data - label.data).numpy()) < 0.1)
            #print(label.data)
            training_accuracy= mean_squared_error(np.ravel(output.data), np.ravel(label.data))#MSEで誤差算出
            #print('MSE Train : %.3f' % training_accuracy)
        #valid
        test_accuracy = 0.0
        for i in range(int(valid_size / batch_size)):
            offset = i * batch_size
            data, label = torch.FloatTensor(valid_x[offset:offset+batch_size]), torch.FloatTensor(valid_t[offset:offset+batch_size])
            output = model(data, None)

            #test_accuracy += np.sum(np.abs((output.data - label.data).numpy()) < 10)
            test_accuracy= mean_squared_error(np.ravel(output.data), np.ravel(label.data))
            
            #print(output.data)
            #print(label.data)
        
        #training_accuracy /= training_size
        #test_accuracy /= valid_size

        print('%d loss: %.3f, training_accuracy: %.5f, valid_accuracy: %.5f' % (
            epoch + 1, running_loss, training_accuracy, test_accuracy))
    
    #test
    test_accuracy = 0.0
    test_x, test_t = mkTestSet(test_size) 
    result=[]
    process=[]
    for i in range(int(test_size / batch_size)):#testではlabelが正解なので、output.data(出力)とlabel.dataを比較する
            offset = i * batch_size
            data, label = torch.FloatTensor(test_x[offset:offset+batch_size]), torch.FloatTensor(test_t[offset:offset+batch_size])
            output = model(data, None)
            test_accuracy= mean_squared_error(np.ravel(output.data), np.ravel(label.data))
            process = output.data.numpy().flatten()
            result.append(process)
    print('%d loss: %.3f, training_accuracy: %.5f, test_accuracy: %.5f' % (
            epoch + 1, running_loss, training_accuracy, test_accuracy))

    #print(test_x)
    #print(type(result))
    #print(result)
    #print(result)
    #data_np=result.numpy()
    data_np = np.asarray(result).flatten()
    #data_np[data_np % 2 == 0]=(data_np[data_np%2==0]+1)*1280/2
    print(len(data_np))
    print(data_np)
    data_np = np.resize(data_np, (test_size*data_length,2))
    #print(data_np[:,([0]+1)*1280/2])
    #submission = pd.Series(data_np) #name=['x','y'])
    #submission.to_csv("C:\\Users\\010170243\\work\\seq2seq\\dataset\\kusakaGomiToCSV\\all\\kusaka_result.csv", header=True, index_label='id')
    
    np.savetxt("C:\\Users\\010170243\\work\\seq2seq\\dataset\\kusakaGomiToCSV\\all\\kusaka_result.csv",            # ファイル名
           X=data_np,                  # 保存したい配列
           delimiter=",",fmt ='%.15f',header="x,y",            # 区切り文字
)


if __name__ == '__main__':
    main()