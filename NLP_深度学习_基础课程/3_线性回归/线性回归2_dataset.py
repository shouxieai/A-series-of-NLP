import numpy as np
import matplotlib.pyplot as plt


class MyDataset:
    def __init__(self,xs,ys,batch_size,shuffle):
        self.xs = xs
        self.ys =ys
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __iter__(self):
        return DataLoader(self)

    def __len__(self):
        return len(self.xs)


class DataLoader:
    def __init__(self,dataset):
        self.dataset = dataset
        self.cursor = 0

        self.indexs = np.arange(len(self.dataset))

        if self.dataset.shuffle:
            np.random.shuffle(self.indexs)


    def __next__(self):
        if self.cursor >= len(self.dataset):
            raise StopIteration

        index = self.indexs[self.cursor:self.cursor + self.dataset.batch_size]


        x = self.dataset.xs[index]
        y = self.dataset.ys[index]

        self.cursor += self.dataset.batch_size

        return x , y




if __name__ == "__main__":
    years = np.array([i for i in range(2000,2022)])
    floors = np.array([i for i in range(23,1,-1)])

    years = (years - 2000) / 22

    prices = np.array([10000,11000,12000,13000,14000,12000,13000,16000,18000,20000,19000,22000,24000,23000,26000,35000,30000,40000,45000,52000,50000,60000])
    prices = prices/60000
    # 数据归一化: 除以最大值, z-score归一化, min-max

    k = 1
    b = 0
    lr = 0.07
    epoch = 5000

    batch_size = 2
    shuffle = True

    dataset = MyDataset(years,prices,batch_size,shuffle)

    for e in range(epoch):
        for year,price in dataset:
            predict = k * year + b
            loss = (predict - price) ** 2

            delta_k =  (k * year + b - price) * year
            delta_b =  (k * year + b - price)

            k -= np.sum(delta_k)/batch_size * lr
            b -= np.sum(delta_b)/batch_size * lr


    while True:
        test_year = (int(input("请输入预测的年份: ")) - 2000) / 22
        predict_price = test_year * k + b
        print(predict_price * 60000)

