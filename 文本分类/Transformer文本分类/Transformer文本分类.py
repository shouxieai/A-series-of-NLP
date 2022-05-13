import os
import numpy as np
import torch
import torch.nn as nn
from  torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from Transformer_Encoder import TransformerEncoder

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
    def __init__(self,text,label,word_2_index,index_2_embedding,max_len):
        self.text = text
        self.label = label
        self.word_2_index = word_2_index
        self.index_2_embedding = index_2_embedding
        self.max_len = max_len


    def __getitem__(self, index):
        text = self.text[index][:self.max_len]
        label = int(self.label[index])
        text_len = len(text)

        word_index = [self.word_2_index.get(i,1) for i in text]
        word_index = word_index + [0] * (self.max_len - len(text))
        word_index = torch.tensor(word_index)
        embedding = self.index_2_embedding(word_index)

        return  embedding,label,text_len



    def __len__(self):
        return len(self.text)


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


class TransformerClassifier(nn.Module):
    def __init__(self,embedding_num,class_num,device="cpu"):
        super().__init__()

        self.transformer = TransformerEncoder(device,embedding_num=embedding_num)
        self.classifier = nn.Linear(embedding_num,class_num)
        self.loss_fun = nn.CrossEntropyLoss()


    def forward(self,batch_embedding,batch_len,batch_label=None):
        out = self.transformer(batch_embedding,batch_len)
        out_ = out[:,0,:]

        pre = self.classifier(out_)

        if batch_label is not None:
            loss = self.loss_fun(pre,batch_label)
            return loss




if __name__ == "__main__":
    train_texts,train_labels = read_data("train",2000)
    dev_texts,dev_labels = read_data("dev",30)

    assert len(train_texts)==len(train_labels)
    assert len(dev_texts) == len(dev_labels)

    embedding_num = 100
    max_len = 30
    batch_size = 200
    epoch = 50
    lr = 0.001
    class_num = len(set(train_labels))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    word_2_index,index_2_embedding = built_curpus(train_texts,embedding_num)

    train_dataset = TextDataset(train_texts,train_labels,word_2_index,index_2_embedding,max_len)
    train_dataloader = DataLoader(train_dataset,batch_size,shuffle=False)

    dev_dataset = TextDataset(dev_texts, dev_labels, word_2_index, index_2_embedding, max_len)
    dev_dataloader = DataLoader(dev_dataset, 2, shuffle=False)

    model = TransformerClassifier(embedding_num,class_num,device)
    model = model.to(device)

    optim = torch.optim.AdamW(model.parameters(),lr = lr)


    for e in range(epoch):
        for batch_embedding,batch_label,batch_len in train_dataloader:
            batch_embedding = batch_embedding.to(device)
            batch_label = batch_label.to(device)
            loss = model(batch_embedding,batch_len,batch_label)
            loss.backward()

            optim.step()
            optim.zero_grad()

        print(f"loss:{loss:.2f}")


