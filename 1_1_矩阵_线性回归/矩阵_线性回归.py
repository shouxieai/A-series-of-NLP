import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle


def get_data(file = "上海二手房价.csv"):
    datas = pd.read_csv(file,names=["y","x1","x2","x3","x4","x5","x6"],skiprows = 1)

    y = datas["y"].values.reshape(-1,1)
    X = datas[[f"x{i}" for i in range(1,7)]].values

    # z-score :  (x - mean_x) / std
    mean_y = np.mean(y)
    std_y = np.std(y)

    mean_X = np.mean(X,axis = 0, keepdims = True)
    std_X =  np.std(X,axis = 0,keepdims = True)

    y = (y - mean_y) / std_y
    X = (X -  mean_X) / std_X

    return X,y,mean_y,std_y,mean_X,std_X


if __name__ == "__main__":
    X,y,mean_y,std_y,mean_X,std_X = get_data()
    K = np.random.random((6,1))

    epoch = 1000
    lr = 0.1
    b = 0

    for e in range(epoch):
        pre = X @ K + b
        loss = np.sum((pre - y)**2)/len(X)

        G = (pre - y ) / len(X)
        delta_K = X.T @ G
        delta_b = np.mean(G)

        K = K - lr * delta_K
        b = b - lr * delta_b

        print(f"loss:{loss:.3f}")

    while True:
        bedroom = (int(input("请输入卧室数量:")))
        ting = (int(input("请输入客厅数量:")))
        wei = (int(input("请输入卫生间数量:")))
        area = (int(input("请输入面积:")))
        floor = (int(input("请输入楼层:")))
        year = (int(input("请输入建成年份:")))

        test_x = (np.array([bedroom, ting, wei, area, floor, year]).reshape(1, -1) - mean_X) / std_X

        p = test_x @ K + b
        print("房价为: ", p * std_y + mean_y)

