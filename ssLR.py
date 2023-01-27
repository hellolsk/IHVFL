from phe import paillier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
import time
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

def normalization(data):
    mu = data.mean(axis=0)
    std = data.std(axis=0)
    return (data - mu) / std


# 加载数据
def load_data(file_name):
    df = pd.read_csv(file_name)
    # diabetes 8*features
    # fg = df.iloc[:, :4].to_numpy()
    # fh = df.iloc[:, 4:-1].to_numpy()
    # breast
    fg = df.iloc[:, :10].to_numpy()
    fh = df.iloc[:, 10:-1].to_numpy()

    fg = normalization(fg)
    fh = normalization(fh)

    # print(features[0])
    ones = np.ones(shape=fg.shape[0])
    # np.c_按行链接矩阵
    fg = np.c_[fg, ones]
    # print("features:", features[0])
    # print('fixed features shape: ', features_g.shape)

    # 随机划分训练集和测试集
    fg_train,fg_test,fh_train,fh_test=train_test_split(fg,fh,test_size=0.3,random_state=1)
    
    # labels minst
    labels = np.squeeze(df.iloc[:, -1].to_numpy().reshape(1, -1))
    # labels = np.squeeze(df.iloc[:, -1:].to_numpy().reshape(1, -1))
    # labels = normalization(labels)
    # 变为1和-1
    # labels = labels*2-1
    labels_train,labels_test = train_test_split(labels,test_size=0.3,random_state=1)
    # print('labels shape: ', labels.shape)
    return fg_test,fg_train, fh_test,fh_train, labels_test,labels_train

# 批量读取数据
def data_iter(batch_size, x1, x2, y):
    num_examples = len(y)
    indices = list(range(num_examples))
    np.random.shuffle(indices)
    for i in range(0, num_examples,batch_size):
        # batch_indices = indices[i:i+batch_size]
        batch_indices = indices[i:i+batch_size]
        yield x1[batch_indices], x2[batch_indices], y[batch_indices]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def compute_loss(X1,X2, y, w1,w2):
    y_hat = sigmoid(np.dot(X1, w1)+np.dot(X2, w2))
    wx = np.dot(X1,w1)+np.dot(X2,w2)
    loss = -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat+1e-5))
    # loss = np.sum(np.log(2)-1/2*y*wx+1/8*wx*wx)
    loss /= len(X1)
    return loss

def compute_gradient(pk,sk,X1,X2, y, w1,w2):
    e = sigmoid(np.dot(X1, w1)+np.dot(X2,w2)) - y
    grad1 = np.dot(X1.T,e)
    pk_e = np.asarray([pk.encrypt(m) for m in e])
    # 随机数没加
    pk_ed = np.asarray([sk.decrypt(m) for m in pk_e])
    # pk_ed = e
    grad2 = np.dot(X2.T, pk_ed)
    grad1 /= len(X1)
    grad2 /= len(X2)
    return grad1,grad2


# 训练
def fit(X1,X2, y,fg_test,fh_test,labels_test):
    print('fit start')
    # 初始化模型参数
    np.random.seed(1)
    losslist=[]
    acclist=[]
    auclist=[]
    w1 = np.ones(X1.shape[1])
    w2 = np.ones(X2.shape[1])
    pk,sk = paillier.generate_paillier_keypair(n_length=1024)
    # 开始训练
    batch_size = 32
    learning_rate = 0.05
    iter_max = 30
    oldloss = 0
    for n_iter in range(1, iter_max+1):
        # compute loss
        loss = compute_loss(X1,X2, y, w1,w2)
        losslist.append(loss)
        # print(f'current loss: {loss}')
        if abs(loss-oldloss) <= 1e-5:
            print(f'loss <= 1e-5, fit finish')
            break
        oldloss = loss
        for (batch_X1,batch_X2, batch_y) in data_iter(batch_size, X1,X2, y):
            grad1,grad2 = compute_gradient(pk,sk,batch_X1,batch_X2, batch_y, w1,w2)
            # print("length of batch_X:",n_iter,"----",batch_X.shape[0])
            w1 -= learning_rate * grad1
            w2 -= learning_rate * grad2
        print("current iter:",n_iter)
        acc,predlist = predict(fg_test, fh_test, labels_test, w1,w2)
        acclist.append(acc)
        fpr, tpr, thresholds = metrics.roc_curve(labels_test,predlist)
        auc = metrics.auc(fpr, tpr)
        # print(auc)
        auclist.append(auc)
        # print("w1:",w1)
        # print("w2:",w2)

    return w1,w2,losslist,acclist,auclist


# 预测
def predict(X1,X2, y, w1,w2):
    count = 0
    pred = sigmoid(np.dot(X1, w1)+np.dot(X2, w2))
    count = sum((pred > 0.5)*1 == y)
    # count = sum((pred > 0.5)*1 == (y+1)/2)
    print("count", count)
    return 100 * count / len(y),pred


if __name__ == '__main__':
    fg_test,fg_train, fh_test,fh_train, labels_test,labels_train = load_data('breast_cancer.csv')
    t1 = time.time()
    w1,w2,losslist,acclist,auclist = fit(fg_train, fh_train,labels_train,fg_test,fh_test,labels_test)
    print("w1:",w1)
    print("w2:",w2)
    print(f'耗时：{time.time()-t1:.3f}s')
    # predict_result = predict(fg_test, fh_test,labels_test, w1,w2)
    # print(f'predict_result: {predict_result}%')

    print("losslist:", losslist)
    print("acclsit:", acclist)
    print("auclist:", auclist)

    plt.plot(np.linspace(0, len(losslist), len(losslist)), losslist)
    plt.ylabel('loss')
    plt.xlabel('iter')
    plt.show()
    plt.plot(np.linspace(0, len(acclist), len(acclist)), acclist)
    plt.ylabel('acc')
    plt.xlabel('iter')
    plt.show()
    plt.plot(np.linspace(0, len(auclist), len(auclist)), auclist)
    plt.ylabel('auc')
    plt.xlabel('iter')
    plt.show()
