from phe import paillier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
import time
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import ass
import ssm

def normalization(data):
    mu = data.mean(axis=0)
    std = data.std(axis=0)
    return (data - mu) / std

def guiyihua(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def load_data(file_name):
    df = pd.read_csv(file_name)
    # diabetes 8*features
    fg = df.iloc[:, :4].to_numpy()
    fh = df.iloc[:, 4:-1].to_numpy()
    # breast
    # fg = df.iloc[:, :10].to_numpy()
    # fh = df.iloc[:, 10:-1].to_numpy()

    fg = normalization(fg)
    fh = normalization(fh)

    # print("fg:",fg)
    ones = np.ones(shape=fg.shape[0])

    fg = np.c_[fg, ones]
    # print("features:", features[0])
    # print('fixed features shape: ', features_g.shape)

    fg_train,fg_test,fh_train,fh_test=train_test_split(fg,fh,test_size=0.3,random_state=1)
    
    labels = np.squeeze(df.iloc[:, -1].to_numpy().reshape(1, -1))
    # labels = np.squeeze(df.iloc[:, -1:].to_numpy().reshape(1, -1))
    # labels = normalization(labels)
    # labels = labels*2-1
    labels_train,labels_test = train_test_split(labels,test_size=0.3,random_state=1)
    # print('labels shape: ', labels.shape)
    return fg_test,fg_train, fh_test,fh_train, labels_test,labels_train


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


def compute_loss(X1,X2, y, w11,w12,w21,w22):
    # w1 = w11+w12
    # w2 = w21+w22
    z = np.dot(X1,w11)+np.dot(X1,w12)+np.dot(X2,w21)+np.dot(X2,w22)
    # y_hat = 0.5+0.125*z
    y_hat = sigmoid(z)
    # print("y_hat:",y_hat)
    loss = -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat+1e-5))
    # loss = np.sum(np.log(2)-1/2*y*wx+1/8*wx*wx)
    loss /= len(X1)
    print("loss:",loss)
    return loss
# x.shape=(batch,n)
def compute_gradient(pk,sk,X1,X2, y, w11,w12,w21,w22):
    z11 = np.dot(X1,w11)
    # print("w11",w11)
    # print("w12",w12)
    # print("w21",w21)
    # print("w22",w22)

    z121,z122 = ssm.ssm(pk,sk,X1,w12)


    z22 = np.dot(X2,w22)
    z211,z212 = ssm.ssm(pk,sk,X2,w21)

    z1 = z11+z121+z211

    z1_e = np.asarray([pk.encrypt(m) for m in z1])
    # z1_e2 = np.asarray([pk.encrypt(m) for m in z1*z1])
    # z1_e3 = np.asarray([pk.encrypt(m) for m in z1*z1*z1])

    z2 = z22+z122+z212
    z = z1_e+z2
    # z_3e = z1_e3+3*z1_e2*z2+3*z1_e*z2*z2+z2*z2*z2
    # y_hat = 0.5+0.197*z-0.004*z_3e
    y_hat = 0.5+0.125*z
    e = y_hat-y

    sk_y_hat = np.asarray([sk.decrypt(m) for m in y_hat])
    y1,y2 = ass.asslist(sk_y_hat,len(sk_y_hat))
    e1 = y1
    e2 = y2-y

    g11 = np.dot(X1.T,e1)
    tmp = np.dot(X1.T,e2)
    g121,g122 = ass.asslist(tmp,len(tmp))


    grad2 = np.dot(X2.T,e)
    sk_g2 = np.asarray([sk.decrypt(m) for m in grad2])
    g21,g22 = ass.asslist(sk_g2,len(sk_g2))
    
    n = len(X1)
    return g11/n,g121/n,g122/n,g21/n,g22/n



def fit(X1,X2, y,fg_test,fh_test,labels_test):
    print('fit start')
    np.random.seed(1)
    losslist=[]
    acclist=[]
    auclist=[]
    w1 = np.ones(X1.shape[1])
    w2 = np.ones(X2.shape[1])

    w11,w12 = ass.asslist(w1,len(w1))
    w21,w22 = ass.asslist(w2,len(w2))
    pk,sk = paillier.generate_paillier_keypair(n_length=1024)

    batch_size = 64
    learning_rate = 0.1
    iter_max = 30
    oldloss = 0
    for n_iter in range(1, iter_max+1):
        # compute loss
        loss = compute_loss(X1,X2, y, w11,w12,w21,w22)
        losslist.append(loss)
        # print(f'current loss: {loss}')
        # if abs(loss-oldloss) <= 1e-5:
        #     print(f'loss <= 1e-5, fit finish')
        #     break
        oldloss = loss
        for (batch_X1,batch_X2, batch_y) in data_iter(batch_size, X1,X2, y):
            g11,g121,g122,g21,g22 = compute_gradient(pk,sk,batch_X1,batch_X2, batch_y, w11,w12,w21,w22)
            # print("length of batch_X:",n_iter,"----",batch_X.shape[0])
            w11 -= learning_rate * (g11+g121)
            w12 -= learning_rate * g122
            w21 -= learning_rate * g21
            w22 -= learning_rate * g22
            # print("w11",w11)
            # print("w12",w12)
            # print("w21",w21)
            # print("w22",w22)

        print("current iter:",n_iter)
        acc,predlist = predict(fg_test, fh_test, labels_test, w11,w12,w21,w22)
        acclist.append(acc)
        fpr, tpr, thresholds = metrics.roc_curve(labels_test,predlist)
        auc = metrics.auc(fpr, tpr)
        # print(auc)
        auclist.append(auc)
        # print("w1:",w1)
        # print("w2:",w2)

    return w11,w12,w21,w22,losslist,acclist,auclist



def predict(X1,X2, y, w11,w12,w21,w22):
    count = 0
    w1 = w11+w12
    w2 = w21+w22
    pred = sigmoid(np.dot(X1, w1)+np.dot(X2, w2))
    count = sum((pred > 0.5)*1 == y)
    # count = sum((pred > 0.5)*1 == (y+1)/2)
    print("count", count)
    return 100 * count / len(y),pred


if __name__ == '__main__':
    fg_test,fg_train, fh_test,fh_train, labels_test,labels_train = load_data('diabetes.csv')
    t1 = time.time()
    w11,w12,w21,w22,losslist,acclist,auclist = fit(fg_train, fh_train,labels_train,fg_test,fh_test,labels_test)
    print("w1:",w11+w12)
    print("w2:",w21+w22)
    print(f'cost：{time.time()-t1:.3f}s')
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
