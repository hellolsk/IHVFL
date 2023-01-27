from phe import paillier 
import numpy as np
import random
import time

# 安全矩阵乘法
# 计算z1 + z2 = 【w】*x
# 放回数值类型，而不是一个元素的数组
def ssm(pk,sk,x,w):
    # w = np.asarray([1., 2., 3.])

    e_w = np.asarray([pk.encrypt(m) for m in w])
    e_wx = np.dot(x , e_w)
    # for y in e_wx:
    # 由于每次只取一个x所以wx是个值，不许迭代解密
    # d_wx = np.asarray([private_key.decrypt(m) for m in e_wx])
    z1 = np.random.rand(len(x))
    z2 = e_wx - z1
    # print(z2)
    z2 = np.asarray([sk.decrypt(m) for m in z2])
    # z2 = np.asarray([sk.decrypt(z2)])
    return z1,z2
   
    # 不支持numpy.int类型
def ssmv(pk,sk,e,x):
    e_e = np.asarray([pk.encrypt(m) for m in e])
    e_g = np.dot(e_e , x)

    # for y in e_wx:
    # 由于每次只取一个x所以wx是个值，不许迭代解密
    # d_wx = np.asarray([private_key.decrypt(m) for m in e_wx])
    z1 = np.random.rand(x.shape[1])
    z2 = e_g- z1
    z2 = np.asarray([sk.decrypt(m) for m in z2])
    # print(z1,z2)
    return z1,z2

if __name__ == '__main__':
    pk, sk = paillier.generate_paillier_keypair(n_length=1024)
    time1 = time.time()
    x = np.random.rand(16,11)
    w = np.ones(x.shape[1])
    z1,z2 = ssm(pk,sk,x,w)
    time2 = time.time()
    print(time2-time1)
    print("z1:",z1)
    print("z2:",z2)
    print("z1+z2:",z1+z2)
