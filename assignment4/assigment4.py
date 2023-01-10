import math
from multiprocessing.dummy import Array
import random
import string
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from a4_utils import *

def gaussfilter(I, g, d):
    I = cv2.filter2D(I,-1, g)
    I = np.array(I)
    I = I.T

    I = cv2.filter2D(I,-1, d)
    I = np.array(I)
    return I.T


def non_max_suppresion(I: np.ndarray):
    U = I.copy()
 
    for y in range(1, len(I)-1):
        for x in range(1,len(I[1])-1):
            max = np.max(I[y-1:y+2, x-1:x+2])

            if (I[y, x] == max):
                U[y-1:y+2, x-1 ] = 0
                U[y-1:y+2, x+1] = 0
                U[y-1, x] = 0
                U[y+1, x] = 0
            else:
                U[y,x] = 0
        # print(U[y-1:y+2, x-1:x+2])
    U[:, 0] = 0
    U[:, -1] = 0
    U[0, :] =0
    U[-1, :] = 0

    return U

def convolution(I: np.ndarray, g, h):
    g = np.flip(g)
    h = np.flip(h)

    I = cv2.filter2D(I, -1, g)
    I = cv2.filter2D(I.T, -1, h)

    return I.T

def hessian_points(I, sigma, t):
    g = gauss(sigma)
    # g = np.array(g)

    d = gaussdx(sigma)
    # d = np.array(d)
    

    Ix = convolution(I, d, g)
    Iy = convolution(I, g, d)
    Ixx = convolution(Ix, d, g)
    Iyy = convolution(Iy, g, d)

    Ixy = convolution(Ix, g, d)
    
    H = Ixx * Iyy - Ixy*Ixy
    # print(H)

    H[H < t] = 0
    H = non_max_suppresion(H)
    # print(np.max(H))
    return H

def visualize(H):
    pos = np.argwhere(H)
    x = pos[:, 1]
    y = pos[:, 0]
    plt.scatter(x, y,color='r', marker='x')


def nal1a():
    I = cv2.cvtColor(cv2.imread('data/test_points.jpg'), cv2.COLOR_BGR2GRAY).astype('float64')/255

    # print(I)

    plt.subplot(2, 3, 1)
    H3 = hessian_points(I, 3, 0.003)
    plt.imshow(H3)
    plt.subplot(2, 3, 2)
    H6 = hessian_points(I, 6, 0.003)
    plt.imshow(H6)
    plt.subplot(2, 3, 3)
    H9 = hessian_points(I, 9, 0.003)
    plt.imshow(H9)



    plt.subplot(2,3,4)
    plt.imshow(I, cmap="gray")
    visualize(H3)
    
    plt.subplot(2,3,5)
    plt.imshow(I, cmap="gray")
    visualize(H6)

    plt.subplot(2,3,6)
    plt.imshow(I, cmap="gray")
    visualize(H9)
    plt.show()


# nal1a()

def correlationMatrix ( I, sigma, t):
    g = gauss(sigma)
    # print(g)
    d = gaussdx(sigma)
    # print(d)
    Ix = convolution(I, d, g)
    Iy = convolution(I, g, d)

    G = gauss(1.6*sigma)

    C11 = convolution(Ix*Ix, G, G) 
    C12 = convolution(Ix*Iy, G, G)
    C22 = convolution(Iy*Iy, G, G)

    detC = C11 * C22 - C12*C12
    traceC = C11+C22
    alpha = 0.06
    C = detC - alpha * traceC *traceC

    L = non_max_suppresion(C)
    L[L < t] = 0
    
    
    return L, C

    

    

def nal1b():
    I = cv2.cvtColor(cv2.imread('data/graf/graf_a.jpg'), cv2.COLOR_BGR2GRAY).astype('float64')/255
    

    # print(I)
    t = 0.000015
    plt.subplot(2, 3, 1)
    C1, H3 = correlationMatrix(I, 3,t)
    plt.imshow(H3)
    plt.subplot(2, 3, 2)
    C2, H6 = correlationMatrix(I, 6, t)
    plt.imshow(H6)
    plt.subplot(2, 3, 3)
    C3, H9 = correlationMatrix(I, 9, t)
    plt.imshow(H9)



    plt.subplot(2,3,4)
    plt.imshow(I, cmap="gray")
    visualize(C1)

    plt.subplot(2,3,5)
    plt.imshow(I, cmap="gray")
    visualize(C2)

    plt.subplot(2,3,6)
    plt.imshow(I, cmap="gray")
    visualize(C3)
    plt.show()

# nal1b()

def find_corespondances(I1, I2, pos1, pos2):
    
    # pos1 = np.argwhere(pts1)
    # pos2 = np.argwhere(pts2)

    D1 = simple_descriptors(I1, pos1[:,0], pos1[:,1])
    D2 = simple_descriptors(I2, pos2[:,0], pos2[:,1])

    pts = []
    for d1 in D1:
        minindex = np.argmin(np.sum(np.square(np.sqrt(d1) - np.sqrt(D2)), -1)) # odstranil zunanji koren 
                        # in deljeno z 2 iz hellingerDistance, saj to ne vpliva na min razdaljo,
                        #ker sta monotoni funkciji
        pts.append(minindex)

    return np.array(pts)        



def nal2a():
    I1 = cv2.cvtColor(cv2.imread('data/graf/graf_a_small.jpg'), cv2.COLOR_BGR2GRAY).astype('float64')/255
    I2 = cv2.cvtColor(cv2.imread('data/graf/graf_b_small.jpg'), cv2.COLOR_BGR2GRAY).astype('float64')/255
    t = 1e-6
    
    fpI1, _ = correlationMatrix(I1, 6, t )
    fpI2, _ = correlationMatrix(I2, 6, t)
    
    pos1 = np.argwhere(fpI1)
    pos2 = np.argwhere(fpI2)

    pts = find_corespondances(I1, I2, pos1, pos2)

    pts1 = np.flip(pos1[range(len(pos1))], -1)
    pts2 = np.flip(pos2[pts], -1)

    display_matches(I1, pts1, I2, pts2)

# nal2a()

def find_matches(I1, I2, pos1, pos2):

    pts1= find_corespondances(I1, I2, pos1, pos2)
    pts2= find_corespondances(I2, I1, pos2, pos1)

    l = [] 
    bL = pts1 # prvi seznam oblika [range(len(pts1)), bL]

    aD = range(len(pts2)) #
    bD = pts2 # drugi seznam oblika [aD, bD]

    for i in range(len(pts1)):
        # print(i,bL[i] ,np.where(aD == bL[i])[0][0], bD[np.where(aD == bL[i])], sep=" " )
        if ( i == bD[np.where(aD == bL[i])]):
            l.append([i, bL[i]])
    
    return np.array(l)
    
    


def nal2b():
    I1 = cv2.cvtColor(cv2.imread('data/graf/graf_a_small.jpg'), cv2.COLOR_BGR2GRAY).astype('float64')/255
    I2 = cv2.cvtColor(cv2.imread('data/graf/graf_b_small.jpg'), cv2.COLOR_BGR2GRAY).astype('float64')/255
    
    t = 1e-6
    fpI1, _ = correlationMatrix(I1, 6, t )
    fpI2, _ = correlationMatrix(I2, 6, t)

    pos1 = np.argwhere(fpI1)
    pos2 = np.argwhere(fpI2)

    pts = find_matches(I1, I2, pos1, pos2)

    pts1 = np.flip(pos1[pts[:, 0] ], -1)
    pts2 = np.flip(pos2[pts[:, 1] ], -1)

    display_matches(I1, pts1, I2, pts2)


# nal2b()


def nal2e():
    vid = cv2.VideoCapture('data/vid.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter( 'out.avi',fourcc, 59.9, (1920, 1080) )

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            # print("konec")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        kp = sift.detect(gray, None)
        frame = cv2.drawKeypoints(gray, kp, frame)

        out.write(frame)
        cv2.imshow('vid', frame)
        cv2.waitKey(10)

    vid.release()
    

# 2e()


def estimate_homography(pts1, pts2):
    A = []
    v1 = np.zeros((len(pts1), 9))
    v1[:, 0] = pts1[:, 0]
    v1[:, 1] = pts1[:, 1]
    v1[:, 2] = 1
    v1[:, 6] = -pts2[:, 0]*pts1[:, 0]
    v1[:, 7] = -pts2[:, 0]*pts1[:, 1]
    v1[:, 8] = -pts2[:, 0]
    v2 = np.zeros((len(pts1), 9))
    v2[:,3]=  pts1[:, 0]
    v2[:, 4] = pts1[:, 1]
    v2[:,5] = 1
    v2[:, 6] = -pts2[:, 1]*pts1[:, 0]
    v2[:, 7] =-pts2[:, 1]*pts1[:, 1]
    v2[:, 8] = -pts2[:, 1]

    A = np.concatenate((v1, v2), axis=0)

    U, S, VT = np.linalg.svd(A)

    h = VT[8, :]
    h = h/VT[8,8]
    h = np.array(h)
    h = h.reshape((3, 3))

    return h


def visualize_perspective(I1: np.ndarray, pts1, pts2, I2):
    H = estimate_homography(pts1, pts2)

    WP = cv2.warpPerspective(I1, H, I1.shape)
    plt.set_cmap('gray')
    plt.subplot(1,2,2)
    plt.title('rotated')
    plt.imshow(WP)
    plt.subplot(1,2,1)
    plt.imshow(I2)
    plt.title('I2')

    plt.show()

def nal3a():

    I1 = cv2.cvtColor(cv2.imread('data/newyork/newyork_b.jpg'), cv2.COLOR_BGR2GRAY).astype('float64')/255
    I2 = cv2.cvtColor(cv2.imread('data/newyork/newyork_a.jpg'), cv2.COLOR_BGR2GRAY).astype('float64')/255
    
    pts = np.loadtxt('data/newyork/newyork.txt')


    pts1 = pts[:, (0,1)]
    pts2 = pts[:, (2, 3)]



    # display_matches(I1, pts1, I2, pts2)
    
    visualize_perspective(I2, pts1, pts2, I1)

    I1 = cv2.cvtColor(cv2.imread('data/graf/graf_a.jpg'), cv2.COLOR_BGR2GRAY).astype('float64')/255
    I2 = cv2.cvtColor(cv2.imread('data/graf/graf_b.jpg'), cv2.COLOR_BGR2GRAY).astype('float64')/255

    pts = np.loadtxt('data/graf/graf.txt')

    pts1 = pts[:, (0,1)]
    pts2 = pts[:, (2, 3)]

    visualize_perspective(I1, pts1, pts2, I2)


# nal3a()

def nal3b():
    I1 = cv2.cvtColor(cv2.imread('data/graf/graf_b.jpg'), cv2.COLOR_BGR2GRAY).astype('float64')/255
    I2 = cv2.cvtColor(cv2.imread('data/graf/graf_a.jpg'), cv2.COLOR_BGR2GRAY).astype('float64')/255
    
    t = 1e-6
    fpI1, _ = correlationMatrix(I1, 6, t )
    fpI2, _ = correlationMatrix(I2, 6, t)

    pos1 = np.argwhere(fpI1)
    pos2 = np.argwhere(fpI2)

    pts = find_matches(I1, I2, pos1, pos2)

    pts1 = np.flip(pos1[pts[:, 0] ], -1) # koordinate I1
    pts2 = np.flip(pos2[pts[:, 1] ], -1) # koordinate I2

    # print(len(pts1))
    l = 10

    n = random.sample(range(len(pts1)), l)

    bestH = estimate_homography(pts1[n], pts2[n])

    zac1 = np.append(pts1, np.ones((len(pts1), 1)), -1)
    zac2 = np.append(pts2, np.ones((len(pts2), 1)), -1)

    zmnozek = np.matmul(zac1, bestH)
    zmnozek = zmnozek / zmnozek[:, 2, None]

    bestE = np.average(np.sqrt(np.sum(((zmnozek - zac2)[:, (0,1)])**2, -1)))

    for i in range(20):
        
        n = random.sample(range(len(pts1)), l)
        H = estimate_homography(pts1[n], pts2[n])

        zac1 = np.append(pts1, np.ones((len(pts1), 1)), -1)
        zac2 = np.append(pts2, np.ones((len(pts2), 1)), -1)

        zmnozek = np.matmul(zac1, H)
        zmnozek = zmnozek / zmnozek[:, 2, None]

        e = np.average(np.sqrt(np.sum(((zmnozek - zac2)[:, (0,1)])**2, -1)))
        
        if ( e < bestE):

            bestH = H
            bestE = e

    
    WP = cv2.warpPerspective(I1, bestH, I1.shape)
    plt.set_cmap('gray')
    plt.subplot(1,2,2)
    plt.title('rotated')
    plt.imshow(WP)
    plt.subplot(1,2,1)
    plt.imshow(I1)
    plt.title('I2')

    plt.show()


nal3b()