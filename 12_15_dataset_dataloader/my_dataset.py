# import random
# list1 = [1,2,3,4,5,6,7]   # 所有的数据 ,  dataset
#
#
# batch_size =  2          #
# epoch =  2              # 轮次
# shuffle = True
#
# for e in range(epoch):
#     if shuffle:
#         random.shuffle(list1)
#     for i in range(0,len(list1),batch_size): # 数据加载的过程, dataloader
#         batch_data = list1[i:i+batch_size]
#         print(batch_data)

import random
class MyDataset:
    def __init__(self,all_datas,batch_size,shuffle=True):
        self.all_datas = all_datas
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cursor = 0

    # python魔术方法:某种场景自动触发的方法

    #
    def __iter__(self):  # 返回一个具有__next__的对象
        if self.shuffle:
            random.shuffle(self.all_datas)
        self.cursor = 0
        return self



    def __next__(self):
        if self.cursor >= len(self.all_datas):
            raise StopIteration

        batch_data = self.all_datas[self.cursor:self.cursor+self.batch_size]
        self.cursor += self.batch_size
        return batch_data


if __name__ == "__main__":

    all_datas = [1,2,3,4,5,6,7]
    batch_size = 2
    shuffle = True
    epoch = 2

    dataset = MyDataset(all_datas,batch_size,shuffle)

    for e in range(epoch):
        for batch_data in dataset:  # 把一个对象放在for上时, 会自动调用这个对象的__iter__,
            print(batch_data)
