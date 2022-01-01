# (x-2)**2 = 0
from tqdm import trange

epoch = 1000
lr = 0.09
x = 5  # 初始值, 凯明初始化, # 何凯明
label = 0


for e in trange(epoch):

    pre = (x - 2) ** 2
    loss = (pre - label) ** 2

    delta_x = 2*(pre - label) * (x - 2)
    x = x - delta_x * lr

print(x)