import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm

def shuffle_lists(list1,list2):
    list3 = list(zip(list1,list2))
    np.random.shuffle(list3)
    list1[:],list2[:] = zip(*list3)
    return list1,list2

def read_data(train_or_dev,num=None):
    with open(os.path.join("data",train_or_dev+".txt"),encoding='utf-8') as f1:
        all_data = f1.read().split("\n")
    labels = []
    texts = []
    for data in all_data:
        if len(data) == 0:
            continue
        t,l = data.split("\t")
        labels.append(l)
        texts.append(t)
    if num == None:
        return texts,labels
    else:
        texts, labels = shuffle_lists(texts, labels)
        return texts[:num],labels[:num]

def build_curpus(train_texts):
    onehot_curpus = {"<PAD>":0,"<UNK>":1}
    for text in train_texts:
        for word in text:
            onehot_curpus[word] = onehot_curpus.get(word,len(onehot_curpus))
    return onehot_curpus,np.eye(len(onehot_curpus),dtype=np.float32)


class Oh_Dataset(Dataset):
    def __init__(self,texts,labels,word_2_index,index_2_onehot,max_len=30):
        self.texts = texts
        self.labels = labels
        self.index_2_onehot = index_2_onehot
        self.max_len = max_len
        self.word_2_index = word_2_index

    def __getitem__(self, index):
        text = self.texts[index]
        label = int( self.labels[index])

        text_index = [self.word_2_index.get(i,1) for i in text][:30] + [0] * (self.max_len - len(text))
        text_matrix = self.index_2_onehot[text_index]

        return text_matrix,label

    def __len__(self):
        return len(self.labels)


class OhClassifier(nn.Module):
    def __init__(self,curpus_len,hidden_num,class_num,max_len):
        super().__init__()
        self.linear1 = nn.Linear(curpus_len,hidden_num)
        self.active = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear2 = nn.Linear(hidden_num*max_len,class_num)

        self.cross_loss = nn.CrossEntropyLoss()

    def forward(self,input_,label=None):
        hidden = self.linear1(input_)
        hidden_act = self.active(hidden)
        hidden_act_f = self.flatten(hidden_act)

        p = self.linear2(hidden_act_f)
        self.pre = torch.argmax(p,dim=-1).detach().cpu()
        if label != None:
            loss = self.cross_loss(p,label)
            return loss

def test_file():
    global model,device

    test_texts, test_labels = read_data("test")
    test_dataset = Oh_Dataset(test_texts, test_labels, word_2_index, index_2_onehot, max_len)
    test_dataloader = DataLoader(test_dataset, 10, shuffle=False)


    all_pre = []
    for t_texts, t_labels in test_dataloader:
        t_texts = t_texts.to(device)

        model(t_texts)
        all_pre.extend(model.pre.numpy().tolist())
    all_pre = [str(i) for i in all_pre]
    right_num = sum([i == j for i,j in zip(all_pre,test_labels)])
    with open(os.path.join("data","test_result.txt"),"w",encoding="utf-8") as f:
        f.write("\n".join(all_pre))
    print(f"test acc = {right_num / len(test_labels) * 100:.2f}%")

if __name__ == "__main__":
    train_texts,train_labels = read_data("train")
    dev_texts,dev_labels = read_data("dev")

    assert len(train_texts) == len(train_labels)
    assert len(dev_texts) == len(dev_labels)

    word_2_index,index_2_onehot = build_curpus(train_texts)

    epoch = 10
    batch_size = 30
    hidden_num = 50
    max_len = 30
    lr = 0.0007

    train_dataset = Oh_Dataset(train_texts,train_labels,word_2_index,index_2_onehot,max_len)
    train_dataloader = DataLoader(train_dataset,batch_size,shuffle=True)

    dev_dataset = Oh_Dataset(dev_texts, dev_labels, word_2_index, index_2_onehot, max_len)
    dev_dataloader = DataLoader(dev_dataset, batch_size, shuffle=False)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = OhClassifier(len(word_2_index),hidden_num,len(set(train_labels)),max_len)
    model = model.to(device)
    optimer = torch.optim.AdamW(model.parameters(),lr=lr)

    for e in range(epoch):
        for texts,labels in tqdm(train_dataloader):
            texts = texts.to(device)
            labels = labels.to(device)
            loss = model(texts,labels)
            loss.backward()
            optimer.step()
            optimer.zero_grad()
            # if index % 100 == 0:
            #     print(f"loss:{loss:.2f}")

        right_num = 0
        for d_texts,d_labels in dev_dataloader:
            d_texts = d_texts.to(device)
            model(d_texts)
            right_num += int(torch.sum(model.pre == d_labels))
        print(f"dev acc = {right_num/len(dev_labels)*100:.2f}%")
    test_file()