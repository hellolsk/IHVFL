import random
import numpy as np

def asslist(y,n):
    r1 = np.random.rand(n)
    # r1 = y/2
    r2 = y-r1
    return r1,r2
def ass(x,n,m):
    x_1 = np.random.random_sample([n,m])
    x_2 = x - x_1
    # print("x1.shape:",x_1.shape)
    # print("x2.shape:", x_2.shape)
    return x_1,x_2

def getAdditiveShares(secret, fieldSize):
    '''Generate N additive shares from 'secret' in finite field of size 'fieldSize'.'''

    # Generate n-1 shares randomly
    shares_1 = random.randrange(fieldSize)
  
    shares_2 = (secret - shares_1) % fieldSize
    print("s1.shape:",shares_1.shape)
    return shares_1,shares_2


def reconstructSecret(x_1,x_2, fieldSize):
    secret = [[0] * m for _ in range(n)]
    '''Regenerate secret from additive shares'''
    for i in range(0,n):
        for j in range(0, m):
            secret[i][j] = (x_1[i][j]+x_2[i][j]) % fieldSize

    return secret

def matrixShares(matrix,n,m,fieldSize):
    x_1 = [[0] * m for _ in range(n)]
    x_2 = [[0] * m for _ in range(n)]
    i = 0
    j = 0
    for x in matrix:
        # print("x",x)
        for y in x:
            # print("y:",y)
            # print("i,j:",i,j)
            x_1[i][j],x_2[i][j] = getAdditiveShares(y, fieldSize)
            j=j+1

        i = i+1
        j = 0
    return x_1,x_2



if __name__ == "__main__":
    # Generating the shares
    # fieldSize = 10000
    # n = 1
    # m = 2
    # secret=np.random.randint(100, 200, (n, m))
    # print("secret:",secret)
    # x_1,x_2 = matrixShares(secret, n,m, fieldSize)
    # print('x_1 :', x_1)
    # print('x_2 :', x_2)

    # print('Reconstructed secret:', reconstructSecret(x_1,x_2, fieldSize))
    # x = np.random.random_sample([3,3])
    # x_1,x_2 = ass(x,3,3)
    # print(x_1+x_2==x)
    y = np.random.rand(10)
    y1,y2 = asslist(y,len(y))
    print(y1+y2==y)

