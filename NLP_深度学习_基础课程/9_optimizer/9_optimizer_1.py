import numpy as np
import struct
import matplotlib.pyplot as plt
import torch.optim

def load_labels(file):  # 加载数据
    with open(file, "rb") as f:
        data = f.read()
    return np.asanyarray(bytearray(data[8:]), dtype=np.int32)


def load_images(file):  # 加载数据
    with open(file, "rb") as f:
        data = f.read()
    magic_number, num_items, rows, cols = struct.unpack(">iiii", data[:16])
    return np.asanyarray(bytearray(data[16:]), dtype=np.uint8).reshape(num_items, -1)


def make_one_hot(labels,class_num=10):
    result = np.zeros((len(labels),class_num))
    for index,lab in enumerate(labels):
        result[index][lab] = 1
    return result

def sigmoid(x):
    x = np.clip(x,-100,10000000)  # 数据溢出
    return 1/(1+np.exp(-x))

def softmax(x):
    # x = np.clip(x, -10000, 100)  # 数据溢出
    max_x = np.max(x,axis=1,keepdims=True)
    ex = np.exp(x-max_x)
    sum_ex = np.sum(ex,axis=1,keepdims=True)
    result = ex/sum_ex
    result = np.clip(result,1e-100,1)
    return  result

def get_datas():
    train_datas = load_images("..\\mnist_data\\train-images.idx3-ubyte") / 255
    train_label = make_one_hot(load_labels("..\\mnist_data\\train-labels.idx1-ubyte"))

    test_datas = load_images("..\\mnist_data\\t10k-images.idx3-ubyte") / 255
    test_label = load_labels("..\\mnist_data\\t10k-labels.idx1-ubyte")
    return train_datas,train_label,test_datas,test_label


class Linear:
    def __init__(self,in_num,out_num):
        self.weight = np.random.normal(0, 1, size=(in_num, out_num))
        self.u = 0.9
        self.vt = 0


    def forward(self,x):
        self.x = x
        return x @ self.weight

    def backward(self,G):
        delta_weight = self.x.T @ G
        delta_x = G @ self.weight.T

        # ------------------------- SGD ------------------------
        # self.weight -= lr * delta_weight  # 优化器的内容 , 梯度下降优化器 SGD

        # ------------------------- MSGD ------------------------
        self.vt = self.u * self.vt - lr * delta_weight
        self.weight = self.weight + self.vt

        return delta_x

    def __call__(self, x):
        return self.forward(x)

class Sigmoid:
    def forward(self,x):
        self.r = sigmoid(x)
        return self.r

    def backward(self,G):
        return G * self.r * (1-self.r)

    def __call__(self, x):
        return self.forward(x)

class Softmax:


    def forward(self,x):
        self.r = softmax(x)
        return self.r

    def backward(self,G):  #G传label
        return (self.r - G)/self.r.shape[0]

    def __call__(self, x):
        return self.forward(x)


class Mymodel:
    def __init__(self,layers):
        self.layers = layers

    def forward(self,x,label=None):
        for layer in self.layers:
            x = layer(x)
        self.x = x
        if label is not None:
            self.label = label
            loss = -np.sum(label * np.log(x)) / x.shape[0]
            return loss

    def backward(self):
        G = self.label
        for layer in self.layers[::-1]:
            G = layer.backward(G)

    def __call__(self, *args):
        return self.forward(*args)



if __name__ == "__main__":
    train_datas, train_label, test_datas, test_label = get_datas()

    epoch = 20
    batch_size = 200
    lr = 0.05
    hidden_num = 256


    model = Mymodel(
        [
            Linear(784, hidden_num),
            Sigmoid(),
            Linear(hidden_num, 10),
            Softmax()
        ]
    )

    batch_times = int(np.ceil(len(train_datas) / batch_size))
    for e in range(epoch):
        for batch_index in range(batch_times):
            # --------- 获取数据  ------------
            x = train_datas[batch_index*batch_size:(batch_index+1)*batch_size]
            batch_label = train_label[batch_index*batch_size:(batch_index+1)*batch_size]

            # --------- forward  ------------
            loss = model(x,batch_label)
            # if batch_index % 100 == 0:
            #     print(f"loss:{loss:.3f}")
            model.backward()

        x = test_datas
        model(x)
        pre = np.argmax(model.x,axis=1)
        acc = np.sum(pre == test_label)/10000
        print(f"{'*'*20} epoch:{e} {'*'*20}\n acc = {acc:.3f}")
