import math
from multiprocessing.dummy import Array
import string
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from a3_utils import *

def g_kernel( sig):
    size = 2*math.ceil(sig*3)+1
    range = np.linspace(-3*sig-.5, 3*sig+.5, size)
    g = (1/(((2*math.pi )**(1/2))*sig)) * (np.exp(-((np.power(range, 2))/(2*(sig**2)))))
    return  g / sum(g)

def g_kerneldx(sig):
    size = 2*math.ceil(sig*3)+1
    range = np.linspace(-3*sig - 0.5, 3*sig + 0.5, size)
    gdx = (-1/(((2*math.pi )**(1/2))*(sig**3)))* range * (np.exp(-((np.power(range, 2))/(2*(sig**2)))))
    return gdx / sum(abs(gdx))

def gaussfilter(I, g, d):
    I = cv2.filter2D(I,-1, g)
    I = np.array(I)
    I = I.T

    I = cv2.filter2D(I,-1, d)
    I = np.array(I)
    return I.T



def nal1c():
    impulse = np.zeros((50,50))
    impulse[25, 25] = 1

    g = g_kernel( 2)
    d = g_kerneldx( 2)
    g = np.array(g)
    d = np.array(d)

    I = cv2.filter2D(impulse,-1, g)
    I = np.array(I)
    a = np.array(cv2.filter2D(I.T, -1, g))
    a = a.T
    b = np.array(cv2.filter2D(I.T, -1, d))
    b = b.T

    I = cv2.filter2D(impulse, -1, d)
    I = np.array(I)
    c = np.array(cv2.filter2D(I.T, -1, g))
    c = c.T

    I = cv2.filter2D(impulse.T, -1, g)
    I = np.array(I)
    dd = cv2.filter2D(I.T, -1, d)

    I = cv2.filter2D(impulse.T, -1, d)
    I = np.array(I)
    e = cv2.filter2D(I.T, -1, g)

    plt.set_cmap("gray")
    plt.subplot(2,3,1)
    plt.imshow(impulse)
    plt.title("impulse")

    plt.subplot(2,3,2)
    plt.imshow(c)
    # plt.imshow(gaussfilter(impulse,d, g))
    plt.title("D, Gt")

    plt.subplot(2,3,3)
    plt.imshow(b)
    # plt.imshow(gaussfilter(impulse, g, d))
    plt
    plt.title("G, Dt")

    plt.subplot(2,3,4)
    plt.imshow(a)
    plt.title("G, Gt")

    plt.subplot(2,3,5)
    plt.imshow(dd)
    # plt.imshow(gaussfilter(impulse, d, g))
    plt.title("Gt, D")

    plt.subplot(2,3,6)
    plt.imshow(e)
    # plt.imshow(gaussfilter(impulse, g, d))
    plt.title("Dt, G")

    plt.show()


# nal1c()

def partdx(I, sig):

    I = np.array(I)
    g = g_kernel(sig)
    d = g_kerneldx(sig)

    I_g = cv2.filter2D(I.T, -1, d)
    I_x = np.array(cv2.filter2D(I_g.T, -1,  g))
    
    I_g = np.array(cv2.filter2D(I, -1, d))
    I_y = np.array(cv2.filter2D(I_g.T, -1,  g))

    return (I_x, I_y.T)

def secondx(I):
    (I_x, I_y) = partdx(I,2)
    (I_xx, I_xy) = partdx(I_x, 2)
    (I_yx, I_yy) = partdx(I_y, 2)

    return(I_xx, I_xy, I_yx, I_yy)

def mag_dir(I, sig):
    (I_x, I_y) = partdx(I, sig)
    return np.sqrt(I_x**2 + I_y**2), np.arctan2(I_y, I_x)

def nal1d():
    I = cv2.cvtColor(cv2.imread('images/museum.jpg'), cv2.COLOR_BGR2GRAY).astype('float64')

    (I_x, I_y) = partdx(I, 2)
    (I_xx, I_xy, I_yx, I_yy) = secondx(I)

    (mag, dir) = mag_dir(I, 2)

    plt.set_cmap("gray")
    plt.subplot(2,4,1)
    plt.imshow(I)
    plt.subplot(2,4,2)
    plt.title("I_x")
    plt.imshow(I_x)

    plt.subplot(2,4,3)
    plt.title("I_y")
    plt.imshow(I_y)

    plt.subplot(2,4,4)
    plt.title("I_mag")
    plt.imshow(mag)

    plt.subplot(2,4,5)
    plt.title("I_xx")
    plt.imshow(I_xx)

    plt.subplot(2,4,6)
    plt.title("I_xy")
    plt.imshow(I_xy)

    plt.subplot(2,4,7)
    plt.title("I_yy")
    plt.imshow(I_yy)

    plt.subplot(2,4,8)
    plt.title("I_dir")
    plt.imshow(dir)
    
    plt.show()


# nal1d()

def findedges(I, sigma, theta):
    mag, _ = mag_dir(I, sigma)
    return np.where(mag > theta, 255, 0)

def nal2a():
    I = cv2.cvtColor(cv2.imread('images/museum.jpg'), cv2.COLOR_BGR2GRAY).astype("float64")
    
    Ie = findedges(I, 1, 40)

    plt.set_cmap("gray")
    plt.imshow(Ie)
    plt.show()

# nal2a()


def non_max_suppresion(I):
    mag, dir = mag_dir(I, 1)
    U = mag.copy()

    for y in range(1, len(I)-1):
        for x in range(1,len(I[1])-1):
            t = mag[y, x]
            
            kot = (dir[y,x]) % np.pi

            if (kot >= 0 and kot < np.pi/4): # glej vodoravno -
                if( t < mag[y, x-1] or t < mag[y, x+1] ):
                    U[y,x] = 0
            if (kot >= np.pi/4 and kot < np.pi/2): # glej postrani /
                if(  t < mag[y + 1, x-1] or t < mag[y-1, x+1]):
                    U[y,x] = 0
            if (kot >= np.pi/2 and kot < 3*np.pi/4): # navpicno |
                if( t < mag[y -1, x] or t < mag[y+1, x]):
                    U[y,x] = 0
            if (kot >= 3*np.pi/4 and kot < np.pi): # glej postrani \
                if( t < mag[y-1, x-1] or t < mag[y+1, x+1]):
                    U[y,x] = 0

    return U


def nal2c():
    I = cv2.cvtColor(cv2.imread('images/museum.jpg'), cv2.COLOR_BGR2GRAY).astype("float64")
    
    I_nms = non_max_suppresion(I)
    I_findedges = findedges(I, 1, 40)

    I_low = np.where(I_nms> 7,1, 0 )
    I_high = np.where(I_nms > 30, 1, 0)

    n, labels, _, _ = cv2.connectedComponentsWithStats(I_low.astype('uint8'), connectivity=8)
    I_hister = np.zeros_like(I_low)

    for i in range(1, n):
        if ( np.max(I_high[labels == i]) > 0): # v blizini je t_high
            I_hister[labels == i] = 1




    plt.set_cmap("gray")
    plt.subplot(2,2,1)
    plt.imshow(I)
    plt.subplot(2,2,2)
    plt.title("findedges, t = 40")
    plt.imshow(I_findedges)
    plt.subplot(2,2,3)
    plt.title("non-maxima suppresion")
    plt.imshow(I_nms)
    plt.subplot(2,2,4)
    plt.title("Hysteriss, t_low = 7, t_high = 30")
    plt.imshow(I_hister)
    plt.show()

nal2c()

def accumulator(bins_theta, bins_ro, x, y):
    acc = np.zeros((bins_ro, bins_theta))


    for i in range(bins_theta):
        theta = i/bins_theta * np.pi - np.pi/2 # theta je element {-pi/2, pi/2}
        ro = x* np.cos(theta) + y * np.sin(theta)
        j = round((ro + bins_ro)/2) 

        if ( j >= 0 and j < bins_ro):
            acc[j][i] +=1

    return acc

def nal3a():

    plt.set_cmap("viridis")
    plt.subplot(2,2,1)
    plt.imshow(accumulator(300,300, 10, 10))
    plt.subplot(2,2,2)
    plt.imshow(accumulator(300,300, 30, 60))
    plt.subplot(2,2,3)
    plt.imshow(accumulator(300,300, 50, 20))
    plt.subplot(2,2,4)
    plt.imshow(accumulator(300,300, 80, 90))

    plt.show()


# nal3a()

def hough_find_lines(I, bins_theta, bins_ro):
    A = np.zeros((bins_ro, bins_theta))

    D = min(len(I), len(I[1]))


    for y in range(len(I)):
        for x in range(len(I[1])):
            if I[y, x] >= 1:
                A = A + accumulator(bins_theta, bins_ro, x, y)
    return A


def nal3b():

    oneline = cv2.cvtColor(cv2.imread('images/oneline.png'), cv2.COLOR_BGR2GRAY).astype('float64')
    rectangle = cv2.cvtColor(cv2.imread('images/rectangle.png'), cv2.COLOR_BGR2GRAY).astype('float64')


    oneline = findedges(oneline, 1, 10).astype('int32')
    rectangle = findedges(rectangle, 1, 10).astype('int32')


    tst = np.zeros((100, 100))
    tst[10, 10] = 1
    tst[10, 20] = 1

    A = hough_find_lines(tst, 300, 300 )


    plt.subplot(1, 3, 1)
    plt.imshow(A)
    plt.subplot(1, 3, 2)
    plt.imshow(hough_find_lines(oneline, 300, 300 ))
    plt.subplot(1, 3, 3)
    plt.imshow(hough_find_lines(rectangle, 300, 300 ))

    plt.show()

nal3b()

def nonmaxima_suppression_box(I):
    I = np.array(I)
    J = I.copy()

    l = I.shape[0]
    w = I.shape[1]

    for y in range(len(I)):
        for x in range(len(I[1])):
            n = [[ (y-1)%l, (x-1)%w], [(y-1)%l, x], [ (y-1)%l, (x+1)%w],
                    [y, (x-1)%w], [y, x], [y, (x+1)%w],
                    [(y+1)%l, (x-1)%w], [ (y+1)%l, x], [ (y+1)%l, (x+1)%w] ]
            for v in n:

                if I[y, x] < I[ v[0], v[1]]:
                    J[y, x] = 0
                # if J[v[0], v[1]] == J[y, x]:
                #     J[v[0], v[1]] = 0

    return J.astype('int32')




def nal3c():
    oneline = cv2.cvtColor(cv2.imread('images/oneline.png'), cv2.COLOR_BGR2GRAY).astype('float64')
    rectangle = cv2.cvtColor(cv2.imread('images/rectangle.png'), cv2.COLOR_BGR2GRAY).astype('float64')


    oneline_e = findedges(oneline, 1, 10).astype('int32')
    rectangle_e = findedges(rectangle, 1, 10).astype('int32')


    tst = np.zeros((100, 100))
    tst[10, 10] = 1
    tst[10, 20] = 1

    A = hough_find_lines(tst, 200, 200 )
    oneline_line = hough_find_lines(oneline_e, 600, 600 )
    rectangle_line = hough_find_lines(rectangle_e, 600, 600 )
    A_surpress = nonmaxima_suppression_box(A)
    oneline_surpress = nonmaxima_suppression_box(oneline_line)
    rectangle_surpress = nonmaxima_suppression_box(rectangle_line)


    plt.subplot(2, 3, 1)
    plt.imshow(A_surpress)
    plt.subplot(2, 3, 2)
    plt.imshow(oneline_surpress)
    plt.subplot(2, 3, 3)
    plt.imshow(rectangle_surpress)

    t = 100
    oneline_t = np.where(oneline_surpress > t, 1, 0)
    rectangle_t = np.where(rectangle_surpress > 200, 1, 0)
    A_t = np.where(A_surpress > 1, 1, 0)
    
    plt.subplot(2,3,4)  
    l =len(tst)
    w = len(tst[1])
    plt.imshow(tst, cmap="gray")
    for y, x in np.argwhere(A_t):
        ro = (y * 2) - 200 
        theta = x/200 * np.pi - np.pi/2
        draw_line(ro, theta, l, w)


    plt.subplot(2,3,5)  
    plt.imshow(oneline, cmap="gray")
    l =len(oneline)
    w = len(oneline[1])
    for y, x in np.argwhere(oneline_t):
        ro = (y * 2) - 600 
        theta = x/600 * np.pi - np.pi/2
        draw_line(ro, theta, l, w)
    
    plt.subplot(2,3,6)  
    l =len(rectangle)
    w = len(rectangle[1])
    plt.imshow(rectangle, cmap="gray")
    for y, x in np.argwhere(rectangle_t):
        ro = (y * 2) - 600 
        theta = x/600 * np.pi - np.pi/2
        draw_line(ro, theta, l, w)
    


    plt.show()




nal3c()

def nal3e():
    brickRGB = cv2.cvtColor(cv2.imread('images/bricks.jpg'), cv2.COLOR_BGR2RGB)
    pierRGB = cv2.cvtColor(cv2.imread('images/pier.jpg'), cv2.COLOR_BGR2RGB)

    brick = cv2.cvtColor(brickRGB, cv2.COLOR_RGB2GRAY).astype('float64')
    pier = cv2.cvtColor(pierRGB, cv2.COLOR_RGB2GRAY).astype('float64')
    

    brick_e = findedges(brick, 1, 30)
    pier_e = findedges(pier, 1, 10)

    brick_l = hough_find_lines(brick_e, 600, 600)
    pier_l = hough_find_lines(pier_e, 600, 600)

    brick_s = nonmaxima_suppression_box(brick_l)
    pier_s = nonmaxima_suppression_box(pier_l)

    brick_sorted = np.argsort(brick_s, axis=None)[-10:]    
    y, x = np.unravel_index(brick_sorted, brick_s.shape)
    b = zip(y, x)
    
        

    pier_sorted = np.argsort(pier_s, axis=None)[-10:]   
    y, x = np.unravel_index(pier_sorted, pier_s.shape)
    p = zip(y, x)
    
    plt.subplot(1,2,1)
    plt.imshow(pierRGB)
    l = len(pier)
    w = len(pier[1])
    for y, x in p:
        ro = (y * 2) - 600 
        theta = x/600 * np.pi - np.pi/2
        draw_line(ro, theta, l, w)
    
    plt.subplot(1,2,2)

    
    plt.imshow(brickRGB)
    l = len(brick)
    w = len(brick[1])
    for y, x in b:
        ro = (y * 2) - 600 
        theta = x/600 * np.pi - np.pi/2
        draw_line(ro, theta, l, w)

    plt.show()


# nal3e()