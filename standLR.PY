from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
import time
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

def normalization(data):
    mu = data.mean(axis=0)
    std = data.std(axis=0)
    return (data - mu) / std

def guiyihua(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def load_data(file_name):
    df = pd.read_csv(file_name)
    print('read csv data shape: ', df.shape)
    # features
    features = df.iloc[:, :-1].to_numpy()
    features = normalization(features)
    ones = np.ones((features.shape[0],1))
    features = np.hstack((features,ones))
    print('features shape: ', features.shape)
    # labels
    labels = np.squeeze(df.iloc[:, -1:].to_numpy().reshape(1, -1))
    print('labels shape: ', labels.shape)
    return features, labels

def prepare_data(data):
    df = pd.read_csv(data)
    X = df.iloc[:, :-1].to_numpy()
    y = np.squeeze(df.iloc[:, -1:].to_numpy().reshape(1, -1))
    mu = X.mean(axis=0)
    std=X.std(axis=0)
    X=(X-mu)/std
    x_ones = np.ones((X.shape[0],1))
    X=np.hstack((X,x_ones))
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
    
    return X_train,X_test,y_train,y_test




def data_iter(batch_size, X, y):
    num_examples = len(X)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = indices[i:i+batch_size]
        # print("---------------batch_indices:",batch_indices)
        yield X[batch_indices], y[batch_indices]


# sigmoid
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_loss(X, y, w):
    y_hat = sigmoid(np.dot(X, w))
    # print("X.shape",X.shape)
    # print("w.shape",w.shape)
    # print("y_hat",y_hat)
    loss = -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat+1e-5))
    loss /= len(X)
    return loss


def compute_gradient(X, y, w):
    grad = np.dot(X.T, sigmoid(np.dot(X, w)) - y)
    grad /= len(X)
    return grad


def fit(X, y,X_test,y_test):
    print('fit start')
    np.random.seed(1)
    w = np.ones(X.shape[1])
    losslist=[]
    acclist=[]
    auclist=[]
    f1list=[]
    batch_size = 32
    learning_rate = 0.05
    iter_max = 30
    oldloss = 0
    for n_iter in range(1, iter_max+1):
        # compute loss
        loss = compute_loss(X, y, w)
        losslist.append(loss)
        # print(f'current loss: {loss}')
        if abs(loss-oldloss) <= 1e-5:
            print(f'loss <= 1e-5, fit finish')
            break
        oldloss = loss
        for (batch_X, batch_y) in data_iter(batch_size, X, y):
            grad = compute_gradient(batch_X, batch_y, w)
            # print("length of batch_X:",n_iter,"----",batch_X.shape[0])
            w -= learning_rate * grad
        acc,predlist = predict(X_test, y_test, w)
        y_pred = []
        for index,val in enumerate(predlist):
            if val>0.5:
                 y_pred.append(1)
            else:
                 y_pred.append(0)
        acclist.append(acc)
        fpr, tpr, thresholds = metrics.roc_curve(y_test,predlist)
        auc = metrics.auc(fpr, tpr)
        f1 = f1_score(y_test,y_pred)
        f1list.append(f1)
        # print(auc)
        auclist.append(auc)
    return w,losslist, acclist, auclist,f1list


def predict(X, y, w):
    count = 0
    predlist=[]
    for index, x in enumerate(X):
        pred = sigmoid(np.dot(x, w))
        predlist.append(pred)
        if y[index] == 1 and pred > 0.5:
            count += 1
        if y[index] == 0 and pred < 0.5:
            count += 1
    return 100 * count / len(X),predlist



if __name__ == '__main__':
    X_train,X_test,y_train,y_test = prepare_data('breast_cancer.csv')
    t1 = time.time()
    w,losslist, acclist, auclist,f1list= fit(X_train, y_train,X_test,y_test)
    print(f'cost：{time.time()-t1:.3f}s')
    print('w', w)

    print("losslist:", losslist)
    print("acclsit:", acclist)
    print("auclist:", auclist)
    print("f1list:", f1list)

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
    plt.plot(np.linspace(0, len(f1list), len(f1list)), f1list)
    plt.ylabel('f1')
    plt.xlabel('iter')
    plt.show()

    # predict_result = predict(X_test, y_test, w)
    # print(f'predict_result: {predict_result}%')
