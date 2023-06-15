# coding='utf-8'
"""t-SNE对手写数字进行可视化"""
from time import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn import datasets
from sklearn.manifold import TSNE


def get_data():
    #digits = datasets.load_digits(n_class=10)
    data=np.loadtxt('./nuswide_onehot_tc10.txt')
    label=np.loadtxt('./nuswide_onehot_tc10.txt')
    # data=np.loadtxt('./dataset_binary.txt',skiprows=52000)
    # label=np.loadtxt('./dataset_label.txt',skiprows=52000)
    label=[np.argmax(one_hot)for one_hot in label]
    label=np.array(label)
    #data = digits.data
    #label = digits.target
    
    # print(data.shape)
    # print(label.shape)
    # print(label[:9])
    # print(label.type())
    
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features


def plot_embedding(data, label):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    # ax = plt.subplot(111) #表示图像的窗口，一行一列，当前位置为1
    c1 = ['#ff264c','#ffbf26','#fc6f29','#26b6ff','#2668ff','#3fe660','#c834f1','#dbff26','#afada0','#ff6f88']

    for i in range(data.shape[0]): #4000

        color = c1[label[i]]
        plt.text(data[i, 0], data[i, 1], '*' ,
                color=c1[label[i]],
                fontdict={'weight': 'bold', 'size': 13}) # 
    
    # plt.xticks([0,1.2,0.2,0.4,0.6,0.8,1.0])
    # plt.yticks([0,1.2,0.2,0.4,0.6,0.8,1.0])
    plt.xticks([0,1.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    plt.yticks([0,1.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    # plt.title(title)
    plt.savefig('./t-sne1.png')
    plt.show()
    return fig


def main():
    data, label, n_samples, n_features = get_data()
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    # t0 = time()
    # data = data[:]
    # label = label[:]
    result = tsne.fit_transform(data)
    fig = plot_embedding(result, label)
    


if __name__ == '__main__':
    main()