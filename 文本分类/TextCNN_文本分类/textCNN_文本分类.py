import os
import numpy as np
import torch
import torch.nn as nn
from  torch.utils.data import Dataset,DataLoader
from tqdm import tqdm

def read_data(train_or_test,num=None):
    with open(os.path.join("..","data",train_or_test + ".txt"),encoding="utf-8") as f:
        all_data = f.read().split("\n")

    texts = []
    labels = []
    for data in all_data:
        if data:
            t,l = data.split("\t")
            texts.append(t)
            labels.append(l)
    if num == None:
        return texts,labels
    else:
        return texts[:num],labels[:num]


def built_curpus(train_texts,embedding_num):
    word_2_index = {"<PAD>":0,"<UNK>":1}
    for text in train_texts:
        for word in text:
            word_2_index[word] = word_2_index.get(word,len(word_2_index))
    return word_2_index,nn.Embedding(len(word_2_index),embedding_num)

class TextDataset(Dataset):
    def __init__(self,all_text,all_label,word_2_index,max_len):
        self.all_text = all_text
        self.all_label = all_label
        self.word_2_index = word_2_index
        self.max_len = max_len

    def __getitem__(self,index):
        text = self.all_text[index][:self.max_len]
        label = int(self.all_label[index])

        text_idx = [self.word_2_index.get(i,1) for i in text]
        text_idx = text_idx + [0] * (self.max_len - len(text_idx))

        text_idx = torch.tensor(text_idx).unsqueeze(dim=0)

        return  text_idx,label


    def __len__(self):
        return len(self.all_text)

class Block(nn.Module):
    def __init__(self,kernel_s,embeddin_num,max_len,hidden_num):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels=1,out_channels=hidden_num,kernel_size=(kernel_s,embeddin_num)) #  1 * 1 * 7 * 5 (batch *  in_channel * len * emb_num )
        self.act = nn.ReLU()
        self.mxp = nn.MaxPool1d(kernel_size=(max_len-kernel_s+1))

    def forward(self,batch_emb): # 1 * 1 * 7 * 5 (batch *  in_channel * len * emb_num )
        c = self.cnn.forward(batch_emb)
        a = self.act.forward(c)
        a = a.squeeze(dim=-1)
        m = self.mxp.forward(a)
        m = m.squeeze(dim=-1)
        return m


class TextCNNModel(nn.Module):
    def __init__(self,emb_matrix,max_len,class_num,hidden_num):
        super().__init__()
        self.emb_num = emb_matrix.weight.shape[1]

        self.block1 = Block(2,self.emb_num,max_len,hidden_num)
        self.block2 = Block(3,self.emb_num,max_len,hidden_num)
        self.block3 = Block(4,self.emb_num,max_len,hidden_num)
        self.block4 = Block(5, self.emb_num, max_len, hidden_num)

        self.emb_matrix = emb_matrix

        self.classifier = nn.Linear(hidden_num*4,class_num)  # 2 * 3
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self,batch_idx,batch_label=None):
        batch_emb = self.emb_matrix(batch_idx)
        b1_result = self.block1.forward(batch_emb)
        b2_result = self.block2.forward(batch_emb)
        b3_result = self.block3.forward(batch_emb)
        b4_result = self.block4.forward(batch_emb)

        feature = torch.cat([b1_result,b2_result,b3_result,b4_result],dim=1) # 1* 6 : [ batch * (3 * 2)]
        pre = self.classifier(feature)

        if batch_label is not None:
            loss = self.loss_fun(pre,batch_label)
            return loss
        else:
            return torch.argmax(pre,dim=-1)


if __name__ == "__main__":
    train_text,train_label = read_data("train")
    dev_text,dev_label = read_data("dev")

    embedding = 50
    max_len = 20
    batch_size = 200
    epoch = 1000
    lr = 0.001
    hidden_num = 2
    class_num = len(set(train_label))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    word_2_index,words_embedding = built_curpus(train_text,embedding)

    train_dataset = TextDataset(train_text,train_label,word_2_index,max_len)
    train_loader = DataLoader(train_dataset,batch_size,shuffle=False)

    dev_dataset = TextDataset(dev_text, dev_label, word_2_index, max_len)
    dev_loader = DataLoader(dev_dataset, batch_size, shuffle=False)


    model = TextCNNModel(words_embedding,max_len,class_num,hidden_num).to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=lr)

    for e in range(epoch):
        for batch_idx,batch_label in train_loader:
            batch_idx = batch_idx.to(device)
            batch_label = batch_label.to(device)
            loss = model.forward(batch_idx,batch_label)
            loss.backward()
            opt.step()
            opt.zero_grad()

        print(f"loss:{loss:.3f}")

        right_num = 0
        for batch_idx,batch_label in dev_loader:
            batch_idx = batch_idx.to(device)
            batch_label = batch_label.to(device)
            pre = model.forward(batch_idx)
            right_num += int(torch.sum(pre==batch_label))

        print(f"acc = {right_num/len(dev_text)*100:.2f}%")
