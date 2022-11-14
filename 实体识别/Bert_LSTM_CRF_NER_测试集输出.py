import os
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer
from transformers import BertModel
from transformers import AdamW
import torch
import torch.nn as nn
from seqeval.metrics import accuracy_score as seq_accuracy_score
from seqeval.metrics import f1_score as seq_f1_score
from seqeval.metrics import precision_score as seq_precision_score
from seqeval.metrics import recall_score as seq_recall_score

# from sklearn.metrics import accuracy_score as sklearn_accuracy_score
# from sklearn.metrics import f1_score as sklearn_f1_score
# from sklearn.metrics import precision_score as sklearn_precision_score
# from sklearn.metrics import recall_score as sklearn_recall_score

def read_data(file):
    with open(file,"r",encoding="utf-8") as f:
        all_data = f.read().split("\n")

    all_text = []
    all_label = []

    text = []
    label = []
    for data in all_data:
        if data == "":
            all_text.append(text)
            all_label.append(label)
            text = []
            label = []
        else:
            t,l = data.split(" ")
            text.append(t)
            label.append(l)

    return all_text,all_label

def build_label(train_label):
    label_2_index = {"PAD":0,"UNK":1}
    for label in train_label:
        for l in label:
            if l not in label_2_index:
                label_2_index[l] = len(label_2_index)
    return label_2_index,list(label_2_index)

class BertDataset(Dataset):
    def __init__(self,all_text,all_label,label_2_index,max_len,tokenizer,is_test=True):
        self.all_text = all_text
        self.all_label = all_label
        self.label_2_index = label_2_index
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

    def __getitem__(self,index):
        if self.is_test:
            self.max_len = len(self.all_label[index])

        text = self.all_text[index]
        label = self.all_label[index][:self.max_len]

        text_index = self.tokenizer.encode(text,add_special_tokens=True,max_length=self.max_len+2,padding="max_length",truncation=True,return_tensors="pt")
        label_index = [0] +  [self.label_2_index.get(l,1) for l in label] + [0] + [0] * (self.max_len - len(text))

        label_index = torch.tensor(label_index)
        return  text_index.reshape(-1),label_index,len(label)

    def __len__(self):
        return self.all_text.__len__()


class Bert_LSTM_NerModel(nn.Module):
    def __init__(self,lstm_hidden,class_num):
        super().__init__()

        self.bert = BertModel.from_pretrained(os.path.join("..","bert_base_chinese"))
        for name,param in self.bert.named_parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(768,lstm_hidden,batch_first=True,num_layers=1,bidirectional=False) # 768 * lstm_hidden
        self.classifier = nn.Linear(lstm_hidden,class_num)

        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self,batch_index,batch_label=None):
        bert_out = self.bert(batch_index)
        bert_out0,bert_out1 = bert_out[0],bert_out[1]# bert_out0:字符级别特征, bert_out1:篇章级别

        lstm_out,_ = self.lstm(bert_out0)

        pre = self.classifier(lstm_out)
        if batch_label is not None:
            loss = self.loss_fun(pre.reshape(-1,pre.shape[-1]),batch_label.reshape(-1))
            return loss
        else:
            return torch.argmax(pre,dim=-1)



if __name__ == "__main__":


    train_text, train_label = read_data(os.path.join("data","train.txt"))
    dev_text,dev_label = read_data(os.path.join("data","dev.txt"))
    test_text,test_label = read_data(os.path.join("data","test.txt"))

    label_2_index,index_2_label = build_label(train_label)

    tokenizer = BertTokenizer.from_pretrained(os.path.join("..","bert_base_chinese"))

    batch_size = 50
    epoch = 100
    max_len = 30
    lr = 0.0005
    lstm_hidden = 128
    do_train = False
    do_test = False
    do_input = True

    device = "cuda:0" if torch.cuda.is_available() else "cpu"




    if do_train:
        train_dataset = BertDataset(train_text,train_label,label_2_index,max_len,tokenizer,is_test=False)
        train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=False)

        dev_dataset = BertDataset(dev_text, dev_label, label_2_index, max_len, tokenizer)
        dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

        model = Bert_LSTM_NerModel(lstm_hidden,len(label_2_index)).to(device)
        opt = AdamW(model.parameters(),lr)

        best_score = -1
        for e in range(epoch):
            model.train()
            for batch_text_index,batch_label_index,batch_len in train_dataloader:
                batch_text_index = batch_text_index.to(device)
                batch_label_index = batch_label_index.to(device)
                loss = model.forward(batch_text_index,batch_label_index)
                loss.backward()

                opt.step()
                opt.zero_grad()

                print(f'loss:{loss:.2f}')


            model.eval()

            all_pre = []
            all_tag = []
            for batch_text_index,batch_label_index,batch_len in dev_dataloader:
                batch_text_index = batch_text_index.to(device)
                batch_label_index = batch_label_index.to(device)
                pre = model.forward(batch_text_index)

                pre = pre.cpu().numpy().tolist()
                tag = batch_label_index.cpu().numpy().tolist()

                for p,t,l in zip(pre,tag,batch_len):
                    p = p[1:1+l]
                    t = t[1:1+l]

                    p = [index_2_label[i] for i in p]
                    t = [index_2_label[i] for i in t]

                    all_pre.append(p)
                    all_tag.append(t)

            f1_score = seq_f1_score(all_tag,all_pre)
            if f1_score > best_score:
                torch.save(model,"best_model.pt")
                best_score = f1_score

            print(f"best_score:{best_score:.2f},f1_score:{f1_score:.2f}")

    if do_test:
        test_dataset = BertDataset(test_text, test_label, label_2_index, max_len, tokenizer,True)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        model = torch.load("best_model.pt")

        all_pre = []
        all_tag = []
        test_out = []
        for idx,(batch_text_index, batch_label_index, batch_len) in enumerate(test_dataloader):
            text = test_text[idx]

            batch_text_index = batch_text_index.to(device)
            batch_label_index = batch_label_index.to(device)
            pre = model.forward(batch_text_index)

            pre = pre.cpu().numpy().tolist()
            tag = batch_label_index.cpu().numpy().tolist()


            pre = pre[0][1:-1]
            tag = tag[0][1:-1]

            pre = [index_2_label[i] for i in pre]
            tag = [index_2_label[i] for i in tag]

            all_pre.append(pre)
            all_tag.append(tag)

            test_out.extend([f"{w} {t}" for w,t in zip(text,pre)])
            test_out.append("")

        f1_score = seq_f1_score(all_tag, all_pre)
        print(f"test_f1_score:{f1_score:.2f}")

        with open("test_out.txt","w",encoding='utf-8') as f:
            f.write("\n".join(test_out))

    if do_input:
        model = torch.load("best_model.pt")
        text = input("请输入：")
        # text = text[:510]
        text_idx = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt")
        text_idx = text_idx.to(device)

        pre = model.forward(text_idx)
        pre = pre[0][1:-1]
        pre = [index_2_label[i] for i in pre]
        print("\n".join([f"{w}：{t}" for w,t in zip(text,pre)]))


