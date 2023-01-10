import math
from multiprocessing.dummy import Array
import random
import string
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from a6_utils import *
import os

def nal1b():
    t = np.loadtxt('data/points.txt')
    
    X = np.copy(t)
    X = X.T
    
    n = len(X[0])
    mean = np.sum(X, -1) / n
    mean = mean.T
    X[0, :] -= mean[0]
    X[1, :] -= mean[1]


    C = np.matmul(X, X.T) / (n - 1)

    U, S , Vt = np.linalg.svd(C)

    eVectors = np.copy(U)
    eVectors[:, 0] *= np.sqrt(S[0])
    eVectors[:, 1] *= np.sqrt(S[1])
    eVectors[0, :] += mean[0]
    eVectors[1, :] += mean[1]

    plt.scatter(t[:, 0], t[:, 1], color= 'b' ,marker= '+')
    plt.scatter(mean[0], mean[1], color = 'r', marker= 'x')
    drawEllipse(mean, C)
    plt.plot([mean[0], eVectors[0, 0]], [mean[1], eVectors[1, 0]], 'r')
    plt.plot([mean[0], eVectors[0, 1]], [mean[1], eVectors[1, 1]], 'g')
    

    plt.show()
   
    # Question: What do you notice about the relationship between the eigenvectors
    #and the data? What happens to the eigenvectors if you change the data or add more
    #points?

    # each eigen vector is a principal component, the length idicates the biggest deviation, 
    # and direction indicates the direction of the variance(in both ways)
    # If for example we add points along the green vector line, the elipsis gets more circular
    # ( for example if we ad points (2, 4), (2, 2), (1, 6) )

    
    

# nal1b()

def nal1d():
    t = np.loadtxt('data/points.txt')
    
    X = np.copy(t)
    X = X.T
    
    n = len(X[0])
    mean = np.sum(X, -1) / n
    mean = mean.T
    X[0, :] -= mean[0]
    X[1, :] -= mean[1]


    C = np.matmul(X, X.T) / (n - 1)

    U, S , Vt = np.linalg.svd(C)
    pdf = S / sum(S) # probability distribution function
    print(pdf)
    cdf = np.cumsum(pdf) # cumulative distribution function
    print(cdf)
    plt.plot([1, 2], cdf)

    plt.show()
    # eigen values: [8.18975683 1.61024317]
    # if we discard all eigen vectors we have a 100% reconstrucion error, thus max recon error = 9,789
    # if we only keep the first vector, the reconstruction error is= 1,1610 / 9,789 = 16,44 %
    # aka. 83,56 % of the variance is explained by the first eigen vector


# nal1d()

def nal1e():
    t = np.loadtxt('data/points.txt')
    
    X = np.copy(t)
    X = X.T
    
    n = len(X[0])
    mean = np.sum(X, -1) / n
    mean = mean.T
    X[0, :] -= mean[0]
    X[1, :] -= mean[1]


    C = np.matmul(X, X.T) / (n - 1)

    U, S , Vt = np.linalg.svd(C)
    Y = [[0,0]]
    for x in X.T:
        y = np.matmul(U.T,x.T )
        Y = np.vstack([Y, y])

    

    Y = np.delete( Y, 0, axis= 0)

    U[:, 1] = 0
    Xq = [[0, 0]]
    for y in Y:
        xq = np.matmul(U, y.T)
        xq[0] += mean[0]
        xq[1] += mean[1]
        Xq = np.vstack([Xq, xq])

    Xq = np.delete(Xq, 0, axis=0)

    plt.scatter(Xq[:, 0], Xq[:, 1])

    plt.show()

    #Question: What happens to the reconstructed points? Where is the data projected to?
    # The points get projected on to the line (or subspace) defined by the remaining vector(s)



# nal1e()

def closestPoint(x, T):
    min = np.argmin(np.linalg.norm(T - x, axis=-1))
    return T[min]


def nal1f():

    t = np.loadtxt('data/points.txt')
    q = np.array([6, 6])
    cPoint = closestPoint(q, t)
    print("The closest point to [6, 6] is: ", cPoint)
    X = np.copy(t)
    X = X.T
    
    n = len(X[0])
    mean = np.sum(X, -1) / n
    mean = mean.T
    X[0, :] -= mean[0]
    X[1, :] -= mean[1]

    q = q - mean

    C = np.matmul(X, X.T) / (n - 1)

    U, S , Vt = np.linalg.svd(C)
    Y = [[0,0]]
    for x in X.T:
        y = np.matmul(U.T,x.T )
        Y = np.vstack([Y, y])

    qPCA = np.matmul(U.T, q.T)

    Y = np.delete( Y, 0, axis= 0)
    
    U[:, 1] = 0
    Xq = [[0, 0]]
    for y in Y:
        xq = np.matmul(U, y.T)
        xq[0] += mean[0]
        xq[1] += mean[1]
        Xq = np.vstack([Xq, xq])

    Xq = np.delete(Xq, 0, axis=0)
    qC = np.matmul(U, qPCA.T)
    qC = qC + mean

    minPoint = closestPoint(qC, Xq)
    indX = np.where(Xq[: ,0] == minPoint[0])[0]
    
    print("Now the closest point is:" , t[indX, :])

# nal1f()

def dualPCA( X):
    n = len(X[0])
    mean = np.sum(X, -1) / n
    X = ( X.T - mean).T
    # print(X)
    # X[0, :] -= mean[0]
    # X[1, :] -= mean[1]

    C = np.matmul(X.T, X) / (n - 1)

    U, S , _ = np.linalg.svd(C)
    S = S + 10**-15
    

    Sn = np.diag(S)
    vmesni = np.sqrt(np.linalg.inv(Sn) / (n-1))
    Uf = np.matmul(X, U)
    Uf = np.matmul(Uf, vmesni)

    print(Uf)
    return Uf, mean


def nal2():
    t = np.loadtxt('data/points.txt')

    X = np.copy(t)
    X = X.T
    
    U, mean = dualPCA(X)
    n = len(X[0])
    X = ( X.T - mean).T
    
    Y = np.empty(shape=[0, n])
    for x in X.T:
        y = np.matmul(U.T,x )
        Y = np.vstack([Y, y])


    Xq = [[0, 0]]
    for y in Y:
        xq = np.matmul(U, y.T)
        xq[0] += mean[0]
        xq[1] += mean[1]
        Xq = np.vstack([Xq, xq])
    Xq = np.delete(Xq, 0, axis=0)
    print(Xq)

    plt.scatter(Xq[:, 0], Xq[:, 1])
    plt.scatter(t[:, 0], t[:,1], color="r", marker="x")

    plt.show()


# nal2()

def reshapeImages(src):
    X = []
    for pic in os.listdir(src):
        source = src + "/" + pic
        I = cv2.cvtColor(cv2.imread(source), cv2.COLOR_BGR2GRAY)
        I = np.reshape(I, -1)
        X.append(I)

    X = np.array(X).T

    return X


def nal3():
    X = reshapeImages("data/faces/1")

    U, mean = dualPCA(X)

    v1 = np.reshape(U[:,0], (96, 84))
    v2 = np.reshape(U[:,1], (96, 84))
    v3 = np.reshape(U[:,2], (96, 84))
    v4 = np.reshape(U[:,3], (96, 84))
    v5 = np.reshape(U[:,4], (96, 84))
    

    plt.subplot(1, 5, 1)
    plt.imshow(v1, cmap="gray")
    plt.subplot(1, 5, 2)
    plt.imshow(v2, cmap="gray")
    plt.subplot(1, 5, 3)
    plt.imshow(v3, cmap="gray")
    plt.subplot(1, 5, 4)
    plt.imshow(v4, cmap="gray")
    plt.subplot(1, 5, 5)
    plt.imshow(v5, cmap="gray")

    plt.show()


    I = cv2.cvtColor(cv2.imread("data/faces/1/001.png"), cv2.COLOR_BGR2GRAY)
    Im = np.copy(I)
    Im = np.reshape(Im, -1)
    Im[4074] = 0

    Y = np.matmul(U.T, Im - mean)
    
    xp = np.matmul(U, Y)
    xp =  xp + mean

    xp = np.reshape(xp, (96, 84))
    plt.set_cmap("gray")
    plt.subplot(1, 3, 1)
    plt.imshow(I)
    plt.title("Original Image")
    plt.subplot(1, 3, 2)
    plt.imshow(np.reshape(Im, (96, 84)), cmap="gray")
    plt.title("One component changed in Image space")
    plt.subplot(1,3,3)
    plt.imshow( xp)
    plt.title("projected back into Image space")
    plt.show()
    
    Y = np.matmul(U.T, np.reshape(I, -1) - mean)
    # print(len(Y))
    Y[0] = 0
    xp = np.matmul(U, Y)
    xp = xp + mean

    plt.subplot(1, 2, 1)
    plt.imshow(I)    
    plt.title("Original")
    plt.subplot(1,2,2)
    plt.imshow(np.reshape(xp, (96, 84)))
    plt.title("component 0 set to 0 in PCA space")


    plt.show()


# nal3()
    # What do the resulting images represent (both
    #numerically and in the context of faces)?

    # the first image is the first eigen vector,
    # this vector represents the highest variance, 
    # it shows the type of lighting that was mose effective?


def nal3c():
    # s tem ko OHRANIŠ samo prvih n lastnosti, je potem,
    # ko množiš s U ( 8064 x 64) se pomnoži samo s prvimi
    # n eigen vektroji, (poštudiraj monženje kot ga dela Emil pri NUM)
    # To efektivno pomeni, ohrani prvih n najpomembnejših lastnosti, obraza
    # lepo se vidi na sliki če gledaš od ena proti 32, in primerjaš z tem kar si
    #  dobil pri prejšni nalogi da so najpomembnejši eigen vektorji,
    #  lepo se vidi katere sence in counturs obraza se dodajajo
    X = reshapeImages("data/faces/1")

    U, mean = dualPCA(X)

    I = cv2.cvtColor(cv2.imread("data/faces/1/001.png"), cv2.COLOR_BGR2GRAY)
    
    Im = np.reshape(I, -1)

    Y = np.matmul(U.T, Im - mean)

    Y[32:64] = 0

    y32 = np.copy(Y)
    Y[16:64] = 0
    y16 = np.copy(Y)
    Y[8:17] = 0
    y8 = np.copy(Y)
    Y[4:9] =0
    y4 = np.copy(Y)
    Y[2:5] = 0
    y2 = np.copy(Y)
    Y[1:3] =0
    y1 = np.copy(Y)


    x32 = np.matmul(U, y32) + mean
    x16 = np.matmul(U, y16) + mean
    x8 = np.matmul(U, y8) + mean
    x4 = np.matmul(U, y4) + mean
    x2 = np.matmul(U, y2) + mean
    x1 = np.matmul(U, y1) + mean

    x1 = np.reshape(x1, (96, 84))
    x2 = np.reshape(x2, (96, 84))
    x4 = np.reshape(x4, (96, 84))
    x8= np.reshape(x8, (96, 84))
    x16= np.reshape(x16, (96, 84))
    x32 = np.reshape(x32, (96, 84))
    

    plt.set_cmap("gray")
    plt.subplot(1, 6, 1)
    plt.imshow(x32)
    plt.title("32")
    
    plt.subplot(1, 6, 2)
    plt.imshow(x16)
    plt.title("16")
    plt.subplot(1, 6, 3)
    plt.imshow(x8)
    plt.title("8")
    plt.subplot(1, 6,4)
    plt.imshow(x4)
    plt.title("4")
    plt.subplot(1, 6, 5)
    plt.imshow(x2)
    plt.title("2")
    plt.subplot(1, 6, 6)
    plt.imshow(x1)
    plt.title("1")


    plt.show()

# nal1b()
# nal1d()
# nal1e()
# nal1f()
# nal2()
nal3()
# nal3c()