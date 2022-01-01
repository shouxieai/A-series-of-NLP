import numpy as np
# z-score,min_max
years = np.array([i for i in range(2000,2022)]) # 年份 2000 ~ 2021
years = (years - 2000) / 22 # batch_normalizer layer_noramlizer
prices = np.array([10000,11000,12000,13000,14000,12000,13000,16000,18000,20000,19000,22000,24000,23000,26000,35000,30000,40000,45000,52000,50000,60000])/60000

epoch = 10000

k = 1
b = 1
lr = 0.01

for e in range(epoch):
    for x,label in zip(years,prices):
        pre = k * x + b
        loss = (pre - label) ** 2

        delta_k = 2 * (k * x + b - label) * x
        delta_b = 2 * (k * x + b - label)

        k = k - delta_k * lr
        b = b - delta_b * lr
print(f"k={k}, b={b}")
while True:
    year = (float(input("请输入年份: ")) - 2000)/ 22
    print("预测房价: ",(k * year + b) * 60000)


