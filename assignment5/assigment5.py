import math
from multiprocessing.dummy import Array
import random
import string
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from a5_utils import *
from a4_utils import *

def nal1b():
    # vse enote so v cm
    pz = range(1, 100)
    f = 0.25
    T = 12
    y = np.divide(T*f,pz)

    plt.plot(pz, y)
    plt.ylabel("disparity in cm", size= 15)
    plt.xlabel("distance from camera in cm", size= 15)
    plt.grid()
    plt.show()


def ncc(X, Y): #normalized cross corellation
   
    xavg = np.mean(X)
    yavg = np.mean(Y)
    NCC = np.divide(np.sum((X - xavg) * (Y -  yavg) ),((np.sqrt( np.sum(np.square(X-xavg)) * np.sum(np.square(Y-yavg)) ))))
  
    return  NCC

def cal_disparity(I1, I2, n, w):
    R = np.zeros((len(I1), len(I1[0])))
    for y in range(n, len(I1) - n + 1):
        for x1 in range(n, len(I1[0]) - n + 1):
            maxx2 = 0
            maxNCC = 0
            X = I1[y-n:y+n-1, x1-n:x1+n-1]
            lower = max(n, x1 - w)
            upper = min(len(I2[0]) - n +1, x1 + w)
            for x2 in range(lower, upper):
                Y = I2[y-n:y+n-1, x2-n:x2+n-1]
                NCC = ncc(X, Y)
                if NCC >= maxNCC:
                    maxNCC = NCC
                    maxx2 = x2
            val = np.abs(x1 - maxx2)
            R[y, x1] = val
            if (val > w ):
                R[y, x1] = w
    
    return R

def nal1d():
    I1 = cv2.cvtColor(cv2.imread('data/disparity/office_left.png'), cv2.COLOR_BGR2GRAY)
    I2 = cv2.cvtColor(cv2.imread('data/disparity/office_right.png'), cv2.COLOR_BGR2GRAY)
    # I1 = I1[150:350, 200:500]
    # I2 = I2[150:350, 200:500]
    I1 = cv2.resize(I1, (300, 200))
    I2 = cv2.resize(I2, (300, 200))

    R = cal_disparity(I1, I2, 5, 35)

    plt.subplot(1,2,1)
    plt.imshow(I1, cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(R, cmap='gray')

    plt.show()



def fundamental_matrix(pts):
    pts1 = pts[:, 0:2]
    pts2 = pts[:, 2:4]
    
    npts1, T1 = normalize_points(pts1)
    npts2, T2 = normalize_points(pts2)

    x1 = npts1[:, 0]
    y1 = npts1[:, 1]
    x2 = npts2[:, 0]
    y2 = npts2[:, 1]

    v1 = np.multiply(x1,x2)
    v2 = np.multiply(x2, y1)
    v3 = x2
    v4 = np.multiply(x1, y2)
    v5 = np.multiply(y1 , y2)
    v6 = y2
    v7 = x1
    v8 = y1
    v9 = np.ones(len(x1))
    
    A = np.vstack((v1,v2,v3,v4,v5,v6,v7,v8,v9)).T

    U, D, Vt = np.linalg.svd(A)
    V = Vt.T
    F = np.reshape(V[:, 8], (3,3))
    U, D, Vt = np.linalg.svd(F)

    D[2] = 0
    D = np.diag(D)
    # print(D)
    F = np.matmul(U, D)
    F = np.matmul(F, Vt)

    
    
    F = np.matmul(T2.T, F)
    F = np.matmul(F, T1)
    
    return  F



def nal2b():
    I1 = cv2.cvtColor(cv2.imread('data/epipolar/house1.jpg'), cv2.COLOR_BGR2GRAY)
    I2 = cv2.cvtColor(cv2.imread('data/epipolar/house2.jpg'), cv2.COLOR_BGR2GRAY)
    pts = np.loadtxt('data/epipolar/house_points.txt')
    F = fundamental_matrix(pts)
    # print(F)

    h1 = I1.shape[0]
    w1 = I1.shape[1]
    h2 = I2.shape[0]
    w2 = I2.shape[1]

    plt.subplot(1,2,1)
    plt.imshow(I1, cmap='gray')
    plt.scatter(pts[:, 0], pts[:, 1], color='r' )
    for i in range(0, len(pts)):
        x = np.append(pts[i, 2:4], 1)
        draw_epiline( np.matmul(F.T, x.T) , h1, w1)
    
    plt.subplot(1,2,2)
    plt.imshow(I2, cmap='gray')
    plt.scatter(pts[:, 2], pts[:, 3], color='r' )

    for i in range(0, len(pts)):
        x = np.append(pts[i, 0:2], 1)
        draw_epiline( np.matmul(F, x.T) , h2, w2)


    plt.show()





def reprojection_error(pts, F):
    pts1 = pts[:, 0:2]
    pts2 = pts[:, 2:4]
    s = []
    for i in range(0, len(pts1)):
        e1 = np.matmul(F, np.append(pts1[i, :], 1).T)
        e2 = np.matmul(F.T, np.append(pts2[i, :], 1).T)

        d1 = np.abs(e1[0] * pts2[i, 0] + e1[1]* pts2[i, 1] + e1[2]) / np.sqrt(e1[0]**2 + e1[1]**2)
        d2 = np.abs(e2[0] * pts1[i, 0] + e2[1]* pts1[i, 1] + e2[2]) / np.sqrt(e2[0]**2 + e2[1]**2)
        s.append(d1)
        s.append(d2)

    return np.average(s)


def nal2c():
    pts = np.loadtxt('data/epipolar/house_points.txt')
    F = fundamental_matrix(pts)

    tstPts = np.array([[85, 233, 67, 219]])


    print(reprojection_error(tstPts, F))
    print(reprojection_error(pts, F))




def f_ransac(pts):
    F = fundamental_matrix(pts)
    bestE = reprojection_error(pts, F)
    bestF = F

    for i in range(1000):
        n = random.sample(range(len(pts)), 8)

        F = fundamental_matrix(pts[n])
        
        e = reprojection_error(pts, F)

        if e < bestE:
            bestF = F


    return bestF



def nal2d():
    I1 = cv2.cvtColor(cv2.imread('data/desk/dsc02638.JPG'), cv2.COLOR_BGR2GRAY)
    I2 = cv2.cvtColor(cv2.imread('data/desk/dsc02639.JPG'), cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures= 200)
    kp1, des1 = sift.detectAndCompute(I1, None)
    # print(kp1)
    kp2, des2 = sift.detectAndCompute(I2, None)

    FLANN_INDEX_KDTREE = 1
    ind_params = dict(algorithm= FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=64)

    flann = cv2.FlannBasedMatcher(ind_params, search_params)


    matches = flann.knnMatch(des1, des2, k=2)

    pts1 = []
    pts2 = []

    for i, (m, n) in enumerate(matches):

        if  m.distance < 0.75 * n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    

    pts = np.vstack((pts1[:, 0], pts1[:, 1], pts2[:, 0], pts2[:, 1])).T
    # pts = np.unique(pts, axis=0)    

    _ , idx1 = np.unique(pts1 , return_index=True, axis=0)
    pts = pts[idx1]

    l1 = pts[:, 2:4]
    _ , idx2 = np.unique(l1 ,return_index=True, axis=0)
    pts = pts[idx2]


    # display_matches(I1, pts[:, 0:2], I2, pts[:, 2:4])

    F = f_ransac(pts)
    Fs = fundamental_matrix(pts)

    print(reprojection_error(pts, F))
    print(reprojection_error(pts, Fs))


    h1 = I1.shape[0]
    w1 = I1.shape[1]
    h2 = I2.shape[0]
    w2 = I2.shape[1]


    plt.subplot(1,2, 1)
    plt.imshow(I1,cmap='gray')
    plt.scatter(pts1[:, 0], pts1[:, 1], color='b')
    
    for i in range(0, len(pts)):
        x = np.append(pts[i, 2:4], 1)
        draw_epiline( np.matmul(F.T, x.T) , h1, w1)

    
    plt.subplot(1,2,2)
    plt.imshow(I2,cmap='gray')
    plt.scatter(pts2[:, 0], pts2[:, 1], color='b')
    for i in range(0, len(pts)):
        x = np.append(pts[i,0:2], 1)
        draw_epiline( np.matmul(F, x.T) , h2, w2)

    plt.show()


# nal2d()

def triangulate( pts, P1, P2):
    pts1 = pts[:, 0:2]
    pts2 = pts[:, 2:4]
    
    ena = np.ones(len(pts))

    pts1 = np.vstack((pts1.T, ena)).T
    pts2 = np.vstack((pts2.T, ena)).T
    pts3d = []

    for i in range(len(pts)):
        v1 = pts1[i]
        v2 = pts2[i]

        vek1 = [[0, -v1[2], v1[1]], 
                [v1[2], 0, -v1[0]],
                [-v1[1], v1[0], 0]]

        vek2 = [[0, -v2[2], v2[1]], 
                [v2[2], 0, -v2[0]],
                [-v2[1], v2[0], 0]]

        D1 = np.matmul(vek1, P1)
        D2 = np.matmul(vek2, P2)

        A = np.vstack((D1[0, :], D1[1, :], D2[0, :], D2[1,:]))

        _ , _, Vt = np.linalg.svd(A)
        V = Vt.T

        X = V[:,3]
        X = X / X[3]
        pts3d.append(X[0:3])

    return pts3d


def nal3a():
    I1 = cv2.cvtColor(cv2.imread('data/epipolar/house1.jpg'), cv2.COLOR_BGR2GRAY)
    I2 = cv2.cvtColor(cv2.imread('data/epipolar/house2.jpg'), cv2.COLOR_BGR2GRAY)
    ptsO = np.loadtxt('data/epipolar/house_points.txt')
    P1 = np.loadtxt('data/epipolar/house1_camera.txt')
    P2 = np.loadtxt('data/epipolar/house2_camera.txt')
    pts = triangulate(ptsO, P1, P2)
    pts = np.array(pts)

    T = [[-1, 0,0], [0,0,-1], [0,1,0]]
    T = np.array(T)

    t = []
    for x in pts:
        t.append(np.matmul(T, x.T))

    t = np.array(t)

    stevilke = range(len(pts))
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)

    ax.imshow(I1, cmap='gray')
    ax.scatter(ptsO[:, 0], ptsO[:,1], color='r')
    for i in stevilke:
        ax.text(ptsO[i, 0], ptsO[i,1], str(i))

    ax = fig.add_subplot(1,3,2)
    ax.imshow(I2, cmap='gray')
    ax.scatter(ptsO[:, 2], ptsO[:,3], color='r')
    for i in stevilke:
        ax.text(ptsO[i, 2], ptsO[i,3], str(i))

    ax = fig.add_subplot(1,3,3,projection='3d')
    ax.scatter(t[:, 0], t[:, 1], t[:, 2], color='r')
    for i in stevilke:
        ax.text(t[i, 0], t[i, 1], t[i, 2], str(i))

    plt.show()

# nal1b()
# nal1d()
# nal2b()
# nal2c()
nal2d()
# nal3a()






