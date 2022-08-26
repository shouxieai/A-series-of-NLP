import numpy as np


class MyDataset:
    def __init__(self,xs,ys,zs,batch_size,shuffle):
        self.xs = xs
        self.ys = ys
        self.zs = zs
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
        z = self.dataset.zs[index]
        self.cursor += self.dataset.batch_size

        return x ,y ,z

years = np.array([i for i in range(2000, 2022)])
years = (years - 2000) / 22

floors = np.array([i for i in range(23,1,-1)])
floors = floors/23


prices = np.array(
    [10000, 11000, 12000, 13000, 14000, 12000, 13000, 16000, 18000, 20000, 19000, 22000, 24000, 23000, 26000, 35000,
     30000, 40000, 45000, 52000, 50000, 60000])
prices = prices / 60000


lr = 0.05
epoch = 10000

k1 = 1
k2 = -1
b = 0
batch_size = 8

dataset = MyDataset(years,floors,prices,batch_size,True)


for e in range(epoch) :
    for year,floor,price in dataset:
        predict = k1 * year  + k2 * floor +  1 * b

        loss = np.sum((predict - price) ** 2)

        delta_k1 = np.sum((predict - price) * year)
        delta_k2 = np.sum((predict - price) * floor)

        delta_b = np.sum((predict - price))

        k1 -= lr * delta_k1
        k2 -= lr * delta_k2
        b -=  lr * delta_b

    if e % 100 == 0:
        print(loss)

while True:

    year = (int(input("请输入年份: ")) - 2000) / 22
    floor = (int(input("请输入楼层:")) / 23)

    p = year * k1 + floor * k2 + b

    print("房价为:" , p *  60000)

