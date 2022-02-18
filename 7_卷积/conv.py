import cv2
import matplotlib.pyplot as plt
import numpy as  np
import os

def get_imgs(path:str):
    img_files = os.listdir(path)
    result = []

    for file in img_files:
        file = os.path.join(path,file)
        img = cv2.imread(file)
        img = cv2.resize(img,(150,150))
        img = img.transpose(2,0,1)
        result.append(img)
    return np.array(result)

def conv(imgs,kernel):
    out_channel,in_channel,kernel_h,kernel_w = kernel.shape
    img_num,_,img_h,img_w = imgs.shape

    c_w = img_w - kernel_w + 1
    c_h = img_h - kernel_h + 1


    A = kernel.reshape(out_channel,-1)
    B = np.zeros((img_num,A.shape[1],c_w*c_h))


    c = 0
    for h in range(c_h):
        for w in range(c_w):
            d = imgs[:,:,h:h+kernel_h,w:w+kernel_w]
            d = d.reshape(img_num,-1)
            B[:,:,c] = d
            c += 1

    result = A @ B
    result = result.reshape(img_num,out_channel,c_h,c_w)
    return result


if __name__ =="__main__":
    kernel = np.array([
        [
            [
                [-1, -2, -3],
                [-1, -2, -3],
                [-1, -10, 1]
            ],
            [
                [0, 3, 3],
                [-1, -2, -3],
                [1, 1, 1]
            ],
            [
                [3, 3, 3],
                [-1, -9, 0],
                [-1, -2, -3]
            ]
        ],
        [
            [
                [1, -1, 0],
                [1, -1, 0],
                [1, -1, 0]
            ],
            [
                [1, -1, 0],
                [1, -1, 0],
                [1, -1, 0]
            ],
            [
                [1, -1, 0],
                [1, -1, 0],
                [1, -1, 0]
            ]
        ],

    ])
    imgs = get_imgs("img")
    result = conv(imgs,kernel)
    for i in result:
        for j in i:
            plt.imshow(j,cmap="gray")
            plt.show()
