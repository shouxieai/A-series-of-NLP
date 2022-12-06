from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch
import torch.nn as nn
import os
import time
import math
from tqdm import tqdm

def get_data(path,num=None):
    all_text = []
    all_label = []
    with open(path,"r",encoding="utf8") as f:
        all_data = f.read().split("\n")
    for data in all_data:
        try:
            if len(data) == 0:
                continue
            data_s = data.split("	")
            if len(data_s) != 2:
                continue
            text,label = data_s
            label = int(label)

        except Exception as e:
            print(e)
        else:
            all_text.append(text)
            all_label.append(int(label))
    if num is None:
        return all_text,all_label
    else:
        return all_text[:num], all_label[:num]

def build_word2index(train_text):
    word_2_index =  {"PAD":0,"UNK":1}
    for text in train_text:
        for word in text:
            if word not in word_2_index:
                word_2_index[word] = len(word_2_index)

    return word_2_index


class TextDataset(Dataset):
    def __init__(self,all_text,all_lable):
        self.all_text = all_text
        self.all_lable = all_lable

    def __getitem__(self, index):
        global word_2_index
        text = self.all_text[index]
        text_index = [word_2_index.get(i,1) for i in text]
        label = self.all_lable[index]
        text_len = len(text)
        return text_index,label,text_len


    def process_batch_batch(self, data):
        global max_len,word_2_index,index_2_embeding
        batch_text = []
        batch_label = []
        batch_len = []

        for d in data:
            batch_text.append(d[0])
            batch_label.append(d[1])
            batch_len.append(d[2])
        min_len = min(batch_len)

        batch_text = [i[:max_len] for i in batch_text]


        batch_text = [i + [0]*(max_len-len(i)) for i in batch_text]
        # batch_emebdding = []
        # for text_idx in batch_text:
        #     text_embdding = []
        #     for idx in text_idx:
        #         word_emb = index_2_embeding[idx]
        #         text_embdding.append(word_emb)
        #     batch_emebdding.append(text_embdding)
        return torch.tensor(batch_text),torch.tensor(batch_label)


    def __len__(self):
        return len(self.all_text)

class Positional(nn.Module):
    def __init__(self,embedding_num,max_len = 3000):
        super().__init__()
        self.position = torch.zeros(size=(max_len,embedding_num),requires_grad=False) #  3000 * embedding

        t = torch.arange(1,max_len+1,dtype=torch.float).unsqueeze(1)
        w_i = 1/(10000**((torch.arange(0,embedding_num,2))/embedding_num))

        w_i_t = w_i*t

        self.position[:,::2] = torch.sin(w_i_t)
        self.position[:,1::2] = torch.cos(w_i_t)


    def forward(self,batch_x): # batch * len * 200
        pos = self.position[:batch_x.shape[1],:]
        pos = pos.unsqueeze(dim=0)
        pos = pos.to(batch_x.device)
        result = batch_x + pos
        return result

class M_Self_Attention(nn.Module):
    def __init__(self,embedding_num,n_heads):
        super(M_Self_Attention, self).__init__()
        self.W_Q = nn.Linear(embedding_num,embedding_num,bias=False)
        self.W_K = nn.Linear(embedding_num,embedding_num,bias=False)
        # self.W_L = nn.Linear(embedding_num,max_len,bias=False)
        self.W_V = nn.Linear(embedding_num,embedding_num,bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.n_heads = n_heads

    def forward(self,x):
        b,l,n = x.shape
        x_ = x.reshape(b, self.n_heads, -1, n)
        Q = self.W_Q(x_) # 查询
        K = self.W_K(x_) # 关键

        V = self.W_V(x_) # 值

        # s = (Q@(K.transpose(-1,-2)) + L) / (math.sqrt(x.shape[-1]/1.0))
        s = (Q@(K.transpose(-1,-2)) ) / (math.sqrt(x.shape[-1]/1.0))
        score = self.softmax(s)
        r = score @ V
        r = r.reshape(b,l,n)
        return r

class Add_Norm(nn.Module):
    def __init__(self,embedding_num):
        super().__init__()
        self.Add = nn.Linear(embedding_num,embedding_num)
        self.Norm = nn.LayerNorm(embedding_num)

    def forward(self,x): # B * Layer * emb
        add_x = self.Add(x)
        norm_x = self.Norm(add_x)
        return norm_x


class Feed_Forward(nn.Module):
    def __init__(self,embedding_num,feed_num):
        super(Feed_Forward, self).__init__()
        self.l1 = nn.Linear(embedding_num,feed_num)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(feed_num,embedding_num)

    def forward(self,x):
        l1_x = self.l1(x)
        r_x = self.relu(l1_x)
        l2_x = self.l2(r_x)
        return l2_x

class Block(nn.Module):
    def __init__(self,embeding_dim,n_heads,feed_num):
        super(Block, self).__init__()
        self.att_layer = M_Self_Attention(embeding_dim, n_heads)
        self.add_norm1 = Add_Norm(embeding_dim)

        self.feed_forward = Feed_Forward(embeding_dim, feed_num)
        self.add_norm2 = Add_Norm(embeding_dim)
        self.n = 100

    def forward(self,x):
        att_x = self.att_layer(x)
        adn_x1 = self.add_norm1(att_x)

        adn_x1 = x + adn_x1  # 残差网络

        ff_x = self.feed_forward(adn_x1)
        adn_x2 = self.add_norm2(ff_x)

        adn_x2 = adn_x1 + adn_x2  # 残差网络

        return adn_x2

class Model(nn.Module):
    def __init__(self,word_size,embeding_dim,class_num,n_heads,feed_num,N):
        super().__init__()
        """
        1. 随机数表示字向量
        2. 预训练字向量 :  使用bert 字向量替换, 使用sougou字向量
        3. 自己基于train_text 训练字向量 
        """
        self.embedding = torch.nn.Embedding(word_size,embeding_dim)
        self.positional = Positional(embeding_dim)
        # 5W~18W 短文本数据

        # self.blocks = nn.ModuleList([Block(embedding_num,n_heads,feed_num)]*N)
        self.blocks = nn.Sequential(*[Block(embedding_num,n_heads,feed_num) for i in range(n)])


        self.linear1 = nn.Linear(embeding_dim,class_num)
        self.loss_fun = nn.CrossEntropyLoss()


    def forward(self,x,label=None):
        x = self.embedding(x)
        x = self.positional(x)

        # for b in self.blocks:
        #     x = b(x)

        x = self.blocks(x)


        pre = self.linear1.forward(x)
        pre = torch.mean(pre,dim=1)
        if label is not None:
            loss = self.loss_fun(pre,label)
            return loss
        else:
            return torch.argmax(pre,dim=-1)

def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构

# word2vec 复现

if __name__ == "__main__":
    same_seeds(1007)
    train_text,train_lable = get_data(os.path.join("..","data","文本分类","train.txt"),70000)
    dev_text,dev_lable = get_data(os.path.join("..","data","文本分类","dev.txt"),10000)
    assert len(train_lable) == len(train_text),"训练数据长度都不一样，你玩冒险呢？"
    assert len(dev_text) == len(dev_lable),"验证数据长度都不一样，你玩冒险呢？"

    embedding_num = 200
    word_2_index = build_word2index(train_text)

    train_batch_size = 50
    max_len = 30
    epoch = 10
    lr = 0.001
    n_heads = 2
    N = 2
    feed_num = int(embedding_num*1.2)
    class_num = len(set(train_lable))

    device = "cuda:0" if  torch.cuda.is_available() else "cpu"
    # device = "cpu"

    train_dataset = TextDataset(train_text,train_lable)
    train_dataloader = DataLoader(train_dataset,batch_size=train_batch_size,shuffle=True,collate_fn=train_dataset.process_batch_batch)

    dev_dataset = TextDataset(dev_text, dev_lable)
    dev_dataloader = DataLoader(dev_dataset, batch_size=10, shuffle=False,collate_fn=dev_dataset.process_batch_batch)


    model = Model(len(word_2_index),embedding_num,class_num,n_heads,feed_num,N).to(device)
    opt = torch.optim.Adam(model.parameters(),lr)
    s_time = time.time()
    for e in range(epoch):
        print("*" * 100)

        for bi,(batch_text,batch_label) in (enumerate(train_dataloader,start=1)):

            batch_text = batch_text.to(device)
            batch_label = batch_label.to(device)
            loss = model.forward(batch_text,batch_label)
            loss.backward()
            opt.step()
            opt.zero_grad()


        print(f"loss:{loss:.2f}")
        e_time = time.time()
        # print(f"cost time :{e_time - s_time:.2f}s")
        s_time = time.time()

        right_num = 0
        for bi,(batch_text,batch_label) in (enumerate(dev_dataloader)):

            batch_text = batch_text.to(device)
            batch_label = batch_label.to(device)
            pre = model.forward(batch_text)
            right_num += int(torch.sum(pre == batch_label))
        print(f"acc:{right_num/len(dev_dataset) * 100:.2f}%")