import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

if __name__ == "__main__":
    # -------------毛发长,腿长
    dogs = np.array([[8.9,12],[9,11],[10,13],[9.9,11.2],[12.2,10.1],[9.8,13],[8.8,11.2]],dtype = np.float32)   # 0
    cats = np.array([[3,4],[5,6],[3.5,5.5],[4.5,5.1],[3.4,4.1],[4.1,5.2],[4.4,4.4]],dtype = np.float32)        # 1

    labels = np.array([0]*7 + [1]* 7,dtype = np.int32).reshape(-1,1)

    X = np.vstack((dogs,cats))

    k = np.random.normal(0,1,size=(2,1))
    b = 0
    epoch = 1000
    lr = 0.05

    for e in range(epoch):
        p = X @ k + b
        pre = sigmoid(p)

        loss = -np.sum(labels * np.log(pre) + (1-labels) * np.log(1-pre))

        G = pre - labels
        delta_k = X.T @ G
        delta_b = np.sum(G)

        k = k - lr * delta_k
        b = b - lr * delta_b
        print(loss)

    while True:
        f1 = float(input('请输入毛发长:'))
        f2 = float(input("请输入腿长:"))

        test_x = np.array([f1,f2]).reshape(1,2)
        p = sigmoid(test_x @ k + b )
        if p >0.5:
            print("类别: 猫")
        else:
            print("类别: 狗")



