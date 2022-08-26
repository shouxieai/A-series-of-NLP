import torch
import torch.nn as nn
import numpy as np
import struct
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F

def load_labels(file):  # 加载数据
    with open(file, "rb") as f:
        data = f.read()
    return np.asanyarray(bytearray(data[8:]), dtype=np.int64)


def load_images(file):  # 加载数据
    with open(file, "rb") as f:
        data = f.read()
    magic_number, num_items, rows, cols = struct.unpack(">iiii", data[:16])
    return np.asanyarray(bytearray(data[16:]), dtype=np.float32).reshape(num_items, -1)


class MnistDataset(Dataset):
    def __init__(self,images,labels):
        self.images = images
        self.labels = labels


    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        return image,label

    def __len__(self):
        return self.labels.__len__()

class MnistModel(nn.Module):
    def __init__(self,hidden_num=784,class_num=10):
        super().__init__()
        self.l1 = nn.Linear(784,hidden_num)
        self.active = nn.ReLU()
        self.l2 = nn.Linear(hidden_num,hidden_num)
        self.active2 = nn.ReLU()
        self.att = nn.Parameter(torch.zeros((28,28),device="cuda:0"))
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self,images,labels = None):
        hid = self.l1(images)
        hid_act = self.active(hid)
        l2_out = self.l2(hid_act)
        act2 = self.active2(l2_out)
        act2 = act2.reshape(act2.shape[0],28,28)
        score = F.softmax( act2 * self.att ,dim=-1)

        pre = l2_out.reshape(-1,28,28) * score
        pre = pre.reshape(pre.shape[0],-1)
        if labels is not None:
            loss = self.loss_fun(pre,labels)
            return  loss
        else:
            return torch.argmax(pre,dim=-1)




if __name__ == "__main__":
    train_datas = load_images("data\\train-images.idx3-ubyte") / 255
    train_label = load_labels("data\\train-labels.idx1-ubyte")

    test_datas = load_images("data\\t10k-images.idx3-ubyte") / 255
    test_label = load_labels("data\\t10k-labels.idx1-ubyte")



    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    epoch = 20
    batch = 100
    lr = 0.001
    hidden_num = 784
    class_num = len(set(train_label))

    train_dataset = MnistDataset(train_datas, train_label)
    train_dataloader = DataLoader(train_dataset,batch,shuffle=False)
    test_dataset = MnistDataset(test_datas, test_label)
    test_dataloader = DataLoader(test_dataset, batch, shuffle=False)

    model = MnistModel(hidden_num,class_num).to(device)
    optim = torch.optim.AdamW(model.parameters(),lr=lr)

    for i in range(epoch):
        model.train()
        for idx,(batch_images,batch_labels) in enumerate(train_dataloader):
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            loss = model.forward(batch_images,batch_labels)
            loss.backward()
            optim.step()
            optim.zero_grad()

        right_num = 0
        for idx,(batch_images,batch_labels) in enumerate(test_dataloader):
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            pre = model.forward(batch_images)
            right_num += int(sum(pre == batch_labels))
        print(f"acc:{right_num/len(test_dataset) * 100}%")


