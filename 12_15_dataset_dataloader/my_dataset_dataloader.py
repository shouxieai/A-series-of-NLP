import random
import numpy as np
class MyDataset:
    def __init__(self,all_datas,batch_size,shuffle=True):
        self.all_datas = all_datas
        self.batch_size = batch_size
        self.shuffle = shuffle


    # python魔术方法:某种场景自动触发的方法

    #
    def __iter__(self):  # 返回一个具有__next__的对象
        # if self.shuffle:
        #     random.shuffle(self.all_datas)
        # self.cursor = 0
        # return self
        return DataLoader(self)

    def __len__(self):
        return len(self.all_datas)


    # def __next__(self):
    #     if self.cursor >= len(self.all_datas):
    #         raise StopIteration
    #
    #     batch_data = self.all_datas[self.cursor:self.cursor+self.batch_size]
    #     self.cursor += self.batch_size
    #     return batch_data


class DataLoader:
    def __init__(self,dataset):
        self.dataset = dataset
        self.indexs = [i for i in range(len(self.dataset))]
        if self.dataset.shuffle == True:
            np.random.shuffle(self.indexs)
        self.cursor = 0

    def __next__(self):
        if self.cursor >= len(self.dataset.all_datas):
            raise StopIteration

        index = self.indexs[self.cursor:self.cursor + self.dataset.batch_size]

        batch_data = self.dataset.all_datas[index]
        self.cursor += self.dataset.batch_size
        return batch_data



if __name__ == "__main__":

    all_datas = np.array([1,2,3,4,5,6,7])
    batch_size = 2
    shuffle = True
    epoch = 2

    dataset = MyDataset(all_datas,batch_size,shuffle)

    for e in range(epoch):
        for batch_data in dataset:  # 把一个对象放在for上时, 会自动调用这个对象的__iter__,
            print(batch_data)
