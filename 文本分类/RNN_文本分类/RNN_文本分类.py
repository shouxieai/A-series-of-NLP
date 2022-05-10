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
    # return word_2_index,np.eye(len(word_2_index),dtype=np.float32)
    # return word_2_index,np.random.normal(0,1,(len(word_2_index),embedding_num)).astype(np.float32)
    return word_2_index,nn.Embedding(len(word_2_index),embedding_num)



class TextDataset(Dataset):
    def __init__(self,texts,labels,word_2_index,index_2_embedding,max_len):
        self.texts = texts
        self.labels = labels
        self.word_2_index = word_2_index
        self.index_2_embedding = index_2_embedding
        self.max_len = max_len

    def __getitem__(self, index):
        # 1. 根据index获取数据
        text = self.texts[index]
        label= int(self.labels[index])

        # 2. 填充裁剪数据长度至max_len
        text = text[:self.max_len] # 裁剪


        # 3. 将 中文文本----> index    -----> onehot 形式
        text_index = [word_2_index.get(i,1) for i in text]   # 中文文本----> index
        text_index = text_index + [0] * (self.max_len - len(text_index)) # 填充

        # text_onehot = self.index_2_embedding[text_index]
        text_onehot = self.index_2_embedding(torch.tensor(text_index))

        return text_onehot,label

    def __len__(self):
        return len(self.labels)


class RNNModel(nn.Module):
    def __init__(self,embedding_num,hidden_num,class_num,max_len):
        super().__init__()
        self.rnn = nn.RNN(embedding_num,hidden_num,num_layers=1,bias=True,batch_first=True,bidirectional=False)

        self.classifier = nn.Linear(max_len*hidden_num,class_num)
        self.cross_loss = nn.CrossEntropyLoss()

    def forward(self,text_embedding,labels=None):
        out,h_n = self.rnn(text_embedding,None)
        out = out.reshape(out.shape[0],-1)
        p = self.classifier(out)

        self.pre = torch.argmax(p,dim=-1).detach().cpu().numpy().tolist()
        if labels is not None:
            loss = self.cross_loss(p,lables)
            return loss


def test_file():
    global model,device,word_2_index, index_2_embedding, max_len

    test_texts, test_labels = read_data("test")

    test_dataset = TextDataset(test_texts, test_labels, word_2_index, index_2_embedding, max_len)
    test_dataloader = DataLoader(test_dataset, 10, shuffle=False)

    result = []
    for text,label in test_dataloader:
        text = text.to(device)
        model(text)
        result.extend(model.pre)
    with open(os.path.join("..","data","test_result.txt"),"w",encoding="utf-8") as f:
        f.write("\n".join([str(i) for i in result]))
    test_acc = sum([i == int(j) for i,j in zip(result,test_labels)]) / len(test_labels)
    print(f"test acc = {test_acc * 100:.2f} % ")
    print("test over")



if __name__ == "__main__":
    train_texts,train_labels = read_data("train",20000)
    dev_texts,dev_labels = read_data("dev")

    assert len(train_texts)==len(train_labels)
    assert len(dev_texts) == len(dev_labels)

    epoch = 5
    batch_size = 60
    max_len = 25
    hidden_num = 300
    embedding_num = 200
    lr = 0.0006

    class_num = len(set(train_labels))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    word_2_index,index_2_embedding = built_curpus(train_texts,embedding_num)

    train_dataset = TextDataset(train_texts, train_labels, word_2_index, index_2_embedding,max_len)
    train_dataloader = DataLoader(train_dataset,batch_size,shuffle=False)

    dev_dataset = TextDataset(dev_texts, dev_labels, word_2_index, index_2_embedding, max_len)
    dev_dataloader = DataLoader(dev_dataset, batch_size, shuffle=False)

    model = RNNModel(embedding_num,hidden_num,class_num,max_len)
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(),lr=lr)


    for e in range(epoch):
        for texts,lables in tqdm(train_dataloader):
            texts = texts.to(device)
            lables = lables.to(device)

            loss = model(texts,lables)
            loss.backward()

            optim.step()
            optim.zero_grad()

        right_num = 0
        for texts,labels in dev_dataloader:
            texts = texts.to(device)
            model(texts)
            right_num += int(sum([i==j for i,j in zip(model.pre,labels)]))
        print(f"dev acc : {right_num/len(dev_labels) * 100 : .2f}%")
    test_file()

