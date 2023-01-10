from cProfile import label
from copy import copy
from math import floor
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from UZ_utils import *



def nalA():
    I = imread('images/umbrellas.jpg')
    imshow(I)

def nalB():

    I = Image.open('images/umbrellas.jpg').convert('RGB')  # PIL image.
    I = np.asarray(I)  # Converting to Numpy array.
    I = np.average(I, 2)
    plt.imshow(I)
    plt.set_cmap('gray')
    plt.show()

def nalC():
    I = imread('images/umbrellas.jpg')
    cutout = I[100:200, 100:200, 0]
    plt.subplot(2, 2, 1)
    plt.imshow(I)
    plt.subplot(2,2,2)
    plt.imshow(cutout)
    plt.title("Red")
    plt.set_cmap('gray')
    plt.subplot(2,2,3)
    plt.imshow(I[100:200, 100:200, 1])
    plt.title("Green")
    plt.set_cmap('gray')
    plt.subplot(2,2,4)
    plt.imshow(I[100:200, 100:200, 2])
    plt.title("Blue")
    plt.set_cmap('gray')
    plt.show()

def nalD():
    I = imread('images/umbrellas.jpg')

    for i in range(100, 200):
        for j in range(200, 400):
            I[i][j] = 1 - I[i][j]


    plt.imshow(I)
    plt.show()

def nalE():
    I = Image.open('images/umbrellas.jpg').convert('L')
    I = np.asarray(I)
    I = I.astype(np.float64)
    J = I / 4
    J = np.floor(J)
    J = J.astype(np.uint8)

    plt.set_cmap('gray')
    plt.subplot(1, 2, 1)
    plt.title('original')
    plt.imshow(I)
    plt.subplot(1,2,2)
    plt.title('reduced')
    plt.imshow(J, vmax=255)
    plt.show()


# EXERCISE 2

def nal2A():
    I = Image.open('images/bird.jpg').convert('L')
    I = np.asarray(I)

    J = copy(I)
    threshold = 70
    J[J < threshold] = 0
    J[J >= threshold] = 1
    
    K = copy(I)
    K = np.where(K < threshold, 0, 1)


    plt.subplot(1,3,1)
    plt.title('Original')
    plt.imshow(I, cmap='gray')
    plt.subplot(1,3,2)
    plt.title('threshold = 70')
    plt.imshow(J, cmap='gray')
    plt.subplot(1,3, 3)
    plt.title('Threshold using where')
    plt.imshow(K, cmap='gray')
    plt.show()


def myhist(I, nr):
    I = np.array(I)
    I = I.reshape(-1)
    H = np.zeros(nr)
    j = 255 / nr
    for i in I:
        l = int(i//j)
        if ( l < nr):
            H[l] = H[l] +1
        else:
            H[nr-1] = H[nr-1] +1
    return H/len(I)

def nal2B():
    I = Image.open('images/bird.jpg').convert('L')
    I = np.asarray(I)

    plt.subplot(2,2,1)
    plt.imshow(I, cmap='gray')
    plt.subplot(2,2,2)
    plt.bar( range(255), myhist(I, 255))
    plt.subplot(2,2,3)
    plt.bar( range(100), myhist(I, 100))
    plt.subplot(2,2,4)
    plt.bar( range(25), myhist(I, 25))
    plt.show()


def myhist2(I, nr):
    I = np.array(I)
    I = I.reshape(-1)
    H = np.zeros(nr)

    min = np.min(I)
    max = np.max(I)
    print(max, min)
    j = (max - min) / nr
    for i in I:
        l = int((i- min)//j)
        if ( l < nr):
            H[l] = H[l] +1
        else:
            H[nr-1] = H[nr-1] +1
    return H/len(I)
    # raje I.shape namesto len(I)

def nal2C():
    I = Image.open('images/bird.jpg').convert('L')
    I = np.asarray(I)

    plt.subplot(3,2,1)
    plt.imshow(I, cmap='gray')
    plt.subplot(3,2,3)
    plt.bar( range(255), myhist2(I, 255))
    plt.subplot(3,2,4)
    plt.bar( range(25), myhist2(I, 25))
    plt.subplot(3,2,5)
    plt.title("histograma nad uint8")
    plt.bar( range(255), myhist(I, 255))
    plt.subplot(3,2,6)
    plt.bar( range(25), myhist(I, 25))
    plt.show()

def nal3a():
    I = Image.open('images/mask.png').convert('RGB')
    I = np.array(I, np.uint8)
    n = 5
    SE = np.ones((n,n), np.uint8) 

    I_eroded = cv2.erode(I, SE)
    I_dilated = cv2.dilate(I, SE)
    plt.subplot(2,4,1)
    plt.imshow(I)
    plt.title('Original')
    plt.subplot(2,4,2)
    plt.title('Eroded, SE = 5x5')
    plt.imshow(I_eroded)
    plt.subplot(2,4,3)
    plt.title('Dilated, SE = 5x5')
    plt.imshow(I_dilated)
    plt.subplot(2,4,4)

    SE = np.ones((2, 2), np.uint8) 
    I_eroded = cv2.erode(I, SE)
    I_dilated = cv2.dilate(I, SE)
    plt.title('Eroded, SE = 2x2')
    plt.imshow(I_eroded)
    plt.subplot(2,4,5)
    plt.title('Dilated, SE = 2x2')
    plt.imshow(I_dilated)
    
    SE = np.ones((5,5), np.uint8) 
    I_eroded = cv2.erode(I, SE)
    I_dilated = cv2.dilate(I, SE)

    I_open = cv2.dilate(I_eroded, SE)
    I_close = cv2.erode(I_dilated, SE)

    plt.subplot(2,4,6)
    plt.title("Open, SE = 5x5")
    plt.imshow(I_open)
    plt.subplot(2,4,7)
    plt.title("Close, SE = 5x5")
    plt.imshow(I_close)
    
    plt.show()

def close(I, SE):
    return cv2.erode(cv2.dilate(I, SE), SE)


def mask ( I):
    SE2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
# close(close(I, SE2), SE1),
    return close(I, SE2)

def nal3b():
    
    I = Image.open('images/bird.jpg').convert('L')
    I = np.asarray(I)

    J = copy(I)
    threshold = 55
    J[J < threshold] = 0
    J[J >= threshold] = 1
    

    plt.set_cmap('gray')
    plt.subplot(1,2,1)
    plt.title('Original')
    plt.imshow(J)
    plt.subplot(1,2,2)
    plt.title('Close')
    plt.imshow(mask(J))

    plt.show()


def immask(I, mask):

    mask = np.expand_dims(mask, 2)
    mask.astype(np.uint8)
    return I*mask
    

def nal3C():
    I = Image.open('images/bird.jpg').convert('RGB')
    J = I.convert('L')
    I = np.asarray(I, np.uint8)
    J = np.asarray(J, np.uint8)
    J = copy(J)

    threshold = 55
    J[J < threshold] = 0
    J[J >= threshold] = 1
    J = mask(J)

    plt.imshow(immask(I, J))
    plt.show()

def nal3d():
    I = Image.open('images/eagle.jpg').convert('RGB')
    J = I.convert('L')
    I = np.asarray(I, np.uint8)
    J = np.asarray(J, np.uint8)
    J = copy(J)

    threshold = 182

    J[J < threshold] = 1
    J[J >= threshold] = 0

    
    J = close(J, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)))
    plt.imshow(immask(I, J))
    plt.set_cmap('gray')
    plt.show()

def nal3e():
    I = Image.open('images/coins.jpg').convert('L')
    I = np.asarray(I, np.uint8)

    I = copy(I)
    I[I < 245] = 1
    I[I >= 245] = 0

    SE1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    I = close(I,SE1)

    I = cv2.dilate(cv2.erode(I, SE1), SE1)
    I = np.asarray(I, np.uint8)

    (numLabels, labels, stats, _) = cv2.connectedComponentsWithStats(I, 4, cv2.CV_32S )
    mask = np.zeros( I.shape, np.uint8)
    
    for i in range(0, numLabels):
        if ( stats[i, cv2.CC_STAT_AREA] <= 700):
            mask = cv2.bitwise_or(mask, np.asarray((labels==i ) , np.uint8))

    plt.set_cmap('gray')
    plt.imshow(mask)
    plt.show()

nal3e()

nalA()
nalB()
nalC()
nalD()
nalE()
nal2A()
nal2B()
nal2C()
nal3a()
nal3b()
nal3C()
nal3d()
nal3e()

# Questions
"""
Q: Why would you use different color maps?
A: Dobimo drugače pobarvano sliko

Q: How is inverting a grayscale value defined for uint8 ?
A: 255 - originalValue

Q: The histograms are usually normalized by dividing the result by the
sum of all cells. Why is that?
A: Zato da poudarimo kontrast v sliki z slabim kontrastom

3a) Q: Based on the results, which order of erosion and dilation operations
produces opening and which closing?
A: erosion(dilation(I)) = Closing
    dilation(erosion(I)) = Opening

3d) Q: Why is the background included in the mask and not the object? How
would you fix that in general? (just inverting the mask if necessary doesn’t count)
A: ker je objekt temnjesši kot pa odzadje. Rešitev: Edge detection in s tem omejiš objekt


"""