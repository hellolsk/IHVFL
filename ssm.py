from phe import paillier 
import numpy as np
import random
import time


def ssm(pk,sk,x,w):

    e_w = np.asarray([pk.encrypt(m) for m in w])
    e_wx = np.dot(x , e_w)
    z1 = np.random.rand(len(x))
    z2 = e_wx - z1
    z2 = np.asarray([sk.decrypt(m) for m in z2])
    return z1,z2
   
def ssmv(pk,sk,e,x):
    e_e = np.asarray([pk.encrypt(m) for m in e])
    e_g = np.dot(e_e , x)
    z1 = np.random.rand(x.shape[1])
    z2 = e_g- z1
    z2 = np.asarray([sk.decrypt(m) for m in z2])
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
