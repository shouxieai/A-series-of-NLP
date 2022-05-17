import torch
import os
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn

from transformers import BertTokenizer
from transformers import BertModel

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

class TextDataset(Dataset):
    def __init__(self,texts,labels,max_len,tokenizer):
        self.texts = texts
        self.labels = labels
        # self.word_2_index = word_2_index
        # self.index_2_embedding = index_2_embedding
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        text = self.texts[index][:self.max_len]
        label= int(self.labels[index])
        t_len = len(text)

        return text,label,t_len

    def __len__(self):
        return len(self.labels)

    def pro_batch_data(self,batch_data):
        batch_text, batch_label, batch_len = zip(*batch_data)


        batch_embedding = []
        for text,label,len_ in zip(batch_text, batch_label, batch_len):

            # text_index = [word_2_index.get(i, 1) for i in text]  # 中文文本----> index
            # text_index = text_index + [0] * (batch_max_len- len(text_index))  # 填充
            # text_embedding = self.index_2_embedding(torch.tensor(text_index))
            text_embedding = self.tokenizer.encode(text,add_special_tokens=True,truncation=True,padding='max_length',max_length=self.max_len+2,return_tensors='pt')
            batch_embedding.append(text_embedding)
        batch_embedding = torch.cat(batch_embedding,dim=0)
        return batch_embedding,torch.tensor(batch_label),batch_len


class BertClassifier(nn.Module):
    def __init__(self,class_num):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert_base_chinese")
        self.classifier = nn.Linear(768,class_num)
        self.loss_fun = nn.CrossEntropyLoss()

        for name, param in self.bert.named_parameters():
            param.requires_grad = False


    def forward(self,batch_x,batch_len,label=None):
        bert_out = self.bert(batch_x,attention_mask=(batch_x>0))
        pre = self.classifier(bert_out[1])

        if label is not None:
            loss = self.loss_fun(pre,label)
            return loss
        else :
            return torch.argmax(pre, dim=-1)


if __name__ == "__main__":
    train_texts, train_labels = read_data("train", 10000)
    dev_texts, dev_labels = read_data("dev")

    assert len(train_texts) == len(train_labels)
    assert len(dev_texts) == len(dev_labels)

    embedding_num = 100
    max_len = 35
    batch_size = 20
    epoch = 250
    class_num = 15
    lr = 0.0001

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    tokenizer = BertTokenizer.from_pretrained("bert_base_chinese")


    train_dataset = TextDataset(train_texts, train_labels, max_len,tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=train_dataset.pro_batch_data)

    dev_dataset = TextDataset(dev_texts, dev_labels, max_len,tokenizer)
    dev_dataloader = DataLoader(dev_dataset, 3, shuffle=False, collate_fn=dev_dataset.pro_batch_data)

    model = BertClassifier(class_num).to(device)
    optim = torch.optim.AdamW(model.parameters())
    for e in range(epoch):
        model.train()
        for batch_index,(batch_x,batch_label,batch_len) in enumerate(train_dataloader):
            batch_x = batch_x.to(device)
            batch_label = batch_label.to(device)
            loss =  model.forward(batch_x,batch_len,batch_label)
            loss.backward()
            optim.step()
            optim.zero_grad()
            if batch_index % 30 == 0:
                print(f"loss:{loss:.2f}")

        model.eval()
        right = 0
        for t_batch_index, (t_batch_x, t_batch_label, t_batch_len) in enumerate(dev_dataloader):
            t_batch_x = t_batch_x.to(device)
            t_batch_label = t_batch_label.to(device)
            pre = model(t_batch_x, t_batch_len)
            right += int(torch.sum(pre == t_batch_label))
        print(f"acc:{right / len(dev_texts) * 100:.3f}% ")