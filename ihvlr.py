import numpy as np
import pandas as pd
import random
import time
import ass
import math
import matplotlib.pyplot as plt
import ssm
from phe import paillier
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split

'''
    log2 = 0.3010299956639812
'''


def normalization(data):
    mu = data.mean(axis=0)
    std = data.std(axis=0)
    return (data - mu) / std


def load_data(file_name):
    df = pd.read_csv(file_name)
    # diabetes 8*features
    # fg = df.iloc[:, :4].to_numpy()
    # fh = df.iloc[:, 4:-1].to_numpy()
    fg = df.iloc[:, :10].to_numpy()
    fh = df.iloc[:, 10:-1].to_numpy()
    print("-----",fg.shape,fh.shape)
    # features_g = df.iloc[:, 1:10].to_numpy()
    # features_h = df.iloc[:, 10:].to_numpy()

    fg = normalization(fg)
    fh = normalization(fh)

    # print(features[0])
    ones = np.ones(shape=fg.shape[0])
    fg = np.c_[fg, ones]
    # print("features:", features[0])
    # print('fixed features shape: ', features_g.shape)

    fg_train,fg_test,fh_train,fh_test=train_test_split(fg,fh,test_size=0.3,random_state=1)
    
    # labels minst
    labels = np.squeeze(df.iloc[:, -1].to_numpy().reshape(1, -1))
    # labels = np.squeeze(df.iloc[:, -1:].to_numpy().reshape(1, -1))
    # labels = normalization(labels)
    labels = labels*2-1
    labels_train,labels_test = train_test_split(labels,test_size=0.3,random_state=1)

    # print('labels shape: ', labels.shape)
    return fg_test,fg_train, fh_test,fh_train, labels_test,labels_train


def ss(xg, xh, y):
    x_c1, x_c2 = ass.ass(xg, xg.shape[0], xg.shape[1])
    x_s1, x_s2 = ass.ass(xh, xh.shape[0], xh.shape[1])
    y1, y2 = ass.asslist(y, len(y))
    return x_c1, x_c2, x_s1, x_s2, y1, y2

def data_iter(batch_size, x_c1, x_c2, x_s1, x_s2, y1, y2):
    num_examples = len(y1)
    indices = list(range(num_examples))
    np.random.shuffle(indices)
    for i in range(0, num_examples,batch_size):
        # batch_indices = indices[i:i+batch_size]
        batch_indices = indices[i:i+batch_size]
        yield x_c1[batch_indices], x_c2[batch_indices], x_s1[batch_indices], x_s2[batch_indices], y1[batch_indices], y2[batch_indices]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_loss(x_c1, x_c2, x_s1, x_s2, y1, y2, wc, ws):
    uc_1 = np.dot(x_c1, wc)

    us_1 = np.dot(x_s1, ws)
    u1 = uc_1+us_1
    uc_2 = np.dot(x_c2, wc)
    us_2 = np.dot(x_s2, ws)

    u2 = uc_2+us_2

    L1 = y1*u1+y1*u2
    L2 = y2*u2+y2*u1

    loss = np.sum(np.log(2)-1/2*(L1+L2)+1/8*(u1+u2)*(u1+u2))
    loss /= len(y1)
    return loss



def compute_gradient(pk,sk,bx_c1, bx_c2, bx_s1, bx_s2, by1, by2, wc, ws):

    uc_1 = np.dot(bx_c1, wc)
    # print("bx_c1.shape:",bx_c1.shape)

    # us_1 = np.dot(bx_s1, ws)
    # u1 = uc_1+us_1

    us_2 = np.dot(bx_s2, ws)
    # uc_2 = np.dot(bx_c2, wc)

    us_11, us_12 = ssm.ssm(pk,sk,bx_s1, ws)
    uc_22, uc_21 = ssm.ssm(pk,sk,bx_c2, wc)
    u1 = uc_1+us_11+uc_21
    u2 = us_2+us_12+uc_22
    # print("u:", u1+u2)
    # print("u1:",u1.shape)
    # print("bx_c1:",bx_c1.shape)
    # print("u1-by1:", (u1-by1).shape)

    gc_11 = np.dot(u1-2*by1, bx_c1)
    # print("gc_11.shape:",gc_11.shape)
    gc_121, gc_122 = ssm.ssmv(pk,sk,u1-2*by1, bx_c2)

    gc_22 = np.dot(u2-2*by2, bx_c2)
    gc_211, gc_212 = ssm.ssmv(pk,sk,u2-2*by2, bx_c1)
    gc_1 = gc_11 + gc_121+gc_211
    gc_2 = gc_22 + gc_122+gc_212
    # print("gc_1.shape:",gc_1.shape)

    gs_11 = np.dot(u2-2*by2, bx_s2)

    gs_211, gs_212 = ssm.ssmv(pk,sk,u2-2*by2, bx_s1)
   
    gs_22 = np.dot(u1-2*by1, bx_s1)
    gs_221, gs_222 = ssm.ssmv(pk,sk,u1-2*by1,bx_s2)
    gs_1 = gs_11+gs_211+gs_221
    gs_2 = gs_22+gs_212+gs_222

    s = 4*len(bx_c1)
    gc_1 /= s
    gs_1 /= s
    gc_2 /= s
    gs_2 /= s
    return gc_1, gc_2, gs_1, gs_2



def fit(pk,sk,x_c1, x_c2, x_s1, x_s2, y1, y2,f_g_test,f_h_test,labels_test):
    print('fit start')

    # np.random.seed(1)
    # wc(202,1)
    # xc_1(11982, 202)
    # uc_1.shape (11982,1)
    # us_1.shape: (11982,1)
    wc = np.ones(x_c1.shape[1])
    ws = np.ones(x_s1.shape[1])
    # print("ws:",ws.shape)
    # wc = np.zeros([202,1])
    # ws = np.zeros([583,1])
    # print("w:", w)

    batch_size = 32
    learning_rate = 0.05
    iter_max = 30
    losslist = []
    acclist = []
    auclist = []
    loss = 0
    for n_iter in range(1, iter_max+1):
        time_losss = time.time()
        # compute loss
        old_loss = loss
        loss = compute_loss(x_c1, x_c2, x_s1, x_s2, y1, y2, wc, ws)
        losslist.append(loss)
        print(f'current loss: {loss}')
        # if abs(loss-old_loss) <= 1e-5:
        #     print(f'loss <= 1e-5, fit finish')
        #     print('current:',n_iter)
        #     break
        # print("y1:",y1.shape)
        for (bx_c1, bx_c2, bx_s1, bx_s2, by1, by2) in data_iter(batch_size, x_c1, x_c2, x_s1, x_s2, y1, y2):
            # print("iter:", i)
            gc_1, gc_2, gs_1, gs_2 = compute_gradient(pk,sk,
                bx_c1, bx_c2, bx_s1, bx_s2, by1, by2, wc, ws)
            # print("gc_1.shape:",gc_1.shape)
            gc = (gc_1+gc_2)
            # print("gc:",gc)
            # print("wc:", wc)
            # print("wc.shape:",wc.shape)
            # print("gc:",gc.shape)
            gs = (gs_1+gs_2)
            # print("grad:", grad)
            wc -= learning_rate*(gc)
            ws -= learning_rate*(gs)
            # wc_1 -= learning_rate * gc_1
            # ws_1 -= learning_rate * gs_1
            # wc_2 -= learning_rate * gc_2
            # ws_2 -= learning_rate * gs_2
        # print("i:", i)
        time_losse = time.time()
        print("each peoch:", time_losse-time_losss)
        print("current iter:",n_iter)
        # xg_test, xh_test, y_test = load_data('mybreast_all.csv')
        acc, pred = predict(f_g_test,f_h_test,labels_test, wc, ws)
        acclist.append(acc)
        fpr, tpr, thresholds = metrics.roc_curve((labels_test+1)/2,pred)
        auc = metrics.auc(fpr, tpr)
        # print(auc)
        auclist.append(auc)


    return wc, ws, losslist, acclist, auclist



def predict(xg_test, xh_test, y_test, wc, ws):
    count = 0
    pred = sigmoid(np.dot(xg_test, wc)+np.dot(xh_test, ws))
    # print("pred:",pred)
    count = sum((pred > 0.5)*1 == (y_test+1)/2)
    # for index,y in enumerate(y_test):
    #     if y_test[index] == 1 and pred[index] > 0.5:
    #         count += 1
    #     if y_test[index] == -1 and pred[index] < 0.5:
    #         count += 1
    return 100 * count / len(xg_test), pred


if __name__ == '__main__':
    np.random.seed(1)
    f_g_test,f_g_train, f_h_test,f_h_train, labels_test,labels_train = load_data('breast_cancer.csv')
    # xg, xh, y = load_data('breast.csv')
    # print("y.shape:", y.shape)
    x_c1, x_c2, x_s1, x_s2, y1, y2 = ss(f_g_train, f_h_train, labels_train)
    # print("x_c1.T",x_c1.T.shape)
    # wc = np.ones(x_c1.shape[1])
    # print("wc",wc.shape)
    # test = np.dot(x_c1,wc)
    # print(test.shape)

    pk, sk = paillier.generate_paillier_keypair(n_length=1024)
    t1 = time.time()
    wc, ws, losslist, acclist, auclist = fit(pk,sk,x_c1, x_c2, x_s1, x_s2, y1, y2,f_g_test,f_h_test,labels_test)
    t2 = time.time()
    print(f'耗时：{t2-t1:.3f}s')
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

    # xg_test, xh_test, y_test = load_data('mnist_test_38.csv')
    # # xg_test, xh_test, y_test = load_data('test_breast.csv')

    # predict_result = predict(xg_test, xh_test, y_test, wc, ws)

    # print(f'predict_result: {predict_result}%')
    # print('wc:', wc)
    # print('ws:', ws)
