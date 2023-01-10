import math
from multiprocessing.dummy import Array
import string
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from a2_utils import *
import os

def myhist3(I, n_bins):
    H = np.zeros((n_bins,n_bins,n_bins))

    k = 255 / n_bins
    for i in I:
        for j in i:
            red = int(j[0] // k)
            green = int(j[1] // k)
            blue = int( j[2] // k)
            
            if( red >= n_bins):
                red = n_bins -1
            if( green >= n_bins):    
                green = n_bins -1
            if( blue >= n_bins):    
                blue = n_bins -1

            H[red, green, blue]+= 1

    return H/np.sum(H)


def nal1a():
    I = cv2.cvtColor(cv2.imread('images\lena.png'), cv2.COLOR_BGR2RGB)
    H = myhist3(I, 8)
    plt.subplot(1,2,1)
    plt.imshow(I)
    plt.subplot(1,2,2)
    plt.bar(range(512), H.flatten())

    plt.show()

# nal1a()


def compareHistograms(H, K):
    H = np.array(H)
    K = np.array(K)
    H = H.flatten()
    K = K.flatten()

    L2 = np.sqrt(np.sum(np.square(H - K)))
    chi = np.sum(np.divide( np.square(H - K), H + K + 1e-10)) /2
    intersection = 1 - np.sum(np.minimum(H, K))
    hell = np.sqrt(np.sum(np.square(np.sqrt(H) - np.sqrt(K))) /2)

    return [L2, chi, intersection, hell]


def nal1b():
    I1 = cv2.cvtColor(cv2.imread('dataset/object_01_1.png'), cv2.COLOR_BGR2RGB)
    I2 = cv2.cvtColor(cv2.imread('dataset/object_02_1.png'), cv2.COLOR_BGR2RGB)

    d = compareHistograms(myhist3(I1, 8), myhist3(I2, 8))[0]
    print(d)

# nal1b()

def nal1c():
    I1 = cv2.cvtColor(cv2.imread('dataset/object_01_1.png'), cv2.COLOR_BGR2RGB)
    I2 = cv2.cvtColor(cv2.imread('dataset/object_02_1.png'), cv2.COLOR_BGR2RGB)
    I3 = cv2.cvtColor(cv2.imread('dataset/object_03_1.png'), cv2.COLOR_BGR2RGB)

    h1 = myhist3(I1, 8)
    h2 = myhist3(I2, 8)
    h3 = myhist3(I3, 8)

    plt.subplot(2, 3, 1)
    plt.imshow(I1)
    plt.subplot(2, 3, 2)
    plt.imshow(I2)
    plt.subplot(2, 3, 3)
    plt.imshow(I3)
    plt.subplot(2, 3, 4)
    plt.bar(range(512), h1.flatten())
    plt.title(round(compareHistograms(h1, h1)[0], 2))
    plt.subplot(2, 3, 5)
    plt.bar(range(512), h2.flatten())
    plt.title(round(compareHistograms(h1, h2)[0], 2))
    plt.subplot(2, 3, 6)
    plt.title(round(compareHistograms(h1, h3)[0], 2))
    plt.bar(range(512), h3.flatten())

    plt.show()

    """
    Q: Which image (object_02_1.png or object_03_1.png) is more similar
to image object_01_1.png considering the L2 distance? How about the other three
distances? We can see that all three histograms contain a strongly expressed component
(one bin has a much higher value than the others). Which color does this
bin represent?
    A: Najbolj podoben je object_03_1.png, saj je L2 radalja najmanjsa med njiman,
    za ostale razdalje podobno pogledaš med katerima histograma je najmanjša razdalja,
    največ je črne barve ( odzadje)
    """

# nal1c()

def getAllHistograms(path, n_bins):
    seznam = []
    for im in os.listdir(path):
        im_name = os.path.join(path, im)
        image = cv2.cvtColor(cv2.imread(im_name), cv2.COLOR_BGR2RGB)
        hist = myhist3(image, n_bins).flatten()
        x = [im_name, image, hist]
        seznam.append(x)

    return seznam

def sort_by(a):
    return a[3]
def sort_by_chi(a):
    return a[4]
def sort_by_inter(a):
    return a[5]
def sort_by_hell(a):
    return a[6]

def nal1d(n_bins):
    Comp = cv2.cvtColor(cv2.imread('dataset/object_05_4.png'), cv2.COLOR_BGR2RGB)

    H = myhist3(Comp, n_bins)
    S = getAllHistograms('dataset', n_bins)

    for i in range(len(S)):
        x = compareHistograms( S[i][2], H)
        S[i] = S[i] + x
        

    S.sort(key=sort_by)

    plt.subplot(5,6, 1)
    for i in range(0, 6):
        plt.subplot(5, 6, i+1)
        plt.imshow(S[i][1])
        plt.subplot(5,6, i + 7)
        plt.bar(range(n_bins * n_bins * n_bins), S[i][2])
        plt.title("L2= " + str(round(S[i][3], 3)))
        
    S.sort(key=sort_by_chi)
    for i in range(0,6):
        plt.subplot(5, 6, i + 13)
        plt.imshow(S[i][1])
        plt.title("Chi = " + str(round(S[i][4], 3)))

    S.sort(key=sort_by_inter)
    for i in range(0,6):
        plt.subplot(5, 6, i + 19)
        plt.imshow(S[i][1])
        plt.title("Inter = " + str(round(S[i][5], 3)))

    S.sort(key=sort_by_hell)
    for i in range(0,6):
        plt.subplot(5, 6, i + 25)
        plt.imshow(S[i][1])
        plt.title("Hell = " + str(round(S[i][6], 3)))

    plt.show()
    """
    Q: Which distance is in your opinion best suited for image retrieval? How
does the retrieved sequence change if you use a different number of bins? Is the
execution time affected by the number of bins?

A: L2
    bolj natanci oz. bolj zbircen
    Da, ker se velikost histograma kubicno povecuje z vecanjem n_bins
    """

# nal1d(8)


def nal1e():

    Comp = cv2.cvtColor(cv2.imread('dataset/object_05_4.png'), cv2.COLOR_BGR2RGB)

    H = myhist3(Comp, 8)
    S = getAllHistograms('dataset', 8)

    for i in range(len(S)):
        x = compareHistograms( S[i][2], H)
        S[i] = S[i] + x


    L = []
    for i in range(0, 120):
        L.append(S[i][3])

    J = np.copy(L)
    J.sort()
    B = L == J[0]
    B1 = L == J[1]
    B2 = L == J[2]
    B3 = L == J[3]
    B4 = L == J[4]
    B = B | B1 | B2 | B3 | B4

    # B = L * B

    L = np.array(L)
    J = np.argsort(L)[:5]
    J = J.astype(int)

    plt.subplot(1,2,1)
    plt.title('Unsorted')
    plt.plot(L)
    plt.scatter(J, L[J], color='red')

    L.sort()

    plt.subplot(1,2,2)
    plt.plot(L[:5], 'ro')
    plt.plot(L)
    plt.title("Sorted by L2 distance")
    
    plt.show()

# nal1e()

def simple_convolution(I, k):
    n = (len(k) - 1)//2

    C = []
    for i in range(0, len(I) - 2*n):
        sum = 0
        for j in range(len(k)):
            sum = sum + k[-j  - 1]*I[i +j]
        C.append(sum)

    return C

def nal2b():
    I = read_data('signal.txt')
    K = read_data('kernel.txt')

    C = simple_convolution(I, K)
    Cv = cv2.filter2D(I,-1, K)

    plt.plot(Cv, 'red')
    plt.plot(C, 'g')
    plt.plot(K, 'orange')
    plt.plot(I)

    plt.show()

# nal2b()

def improved_convolution(I, k):
    n = (len(k) - 1)//2

    C = []
    for i in range(len(I)):
        sum = 0
        for j in range(len(k)):
            if ( (i -n +j) >= 0 and (i -n +j) < len(I)):
                sum = sum + k[len(k) -j  - 1]*I[i - n +j]
                # ce je i - n + j < 0 pomeni,
                #  da smo se zunaj I in tam imamo zero-padding.
        C.append(sum)

    return C

def nal2c():
    I = read_data('signal1.txt') # petnajst enk
    K = read_data('kernel1.txt') # pet enk

    C = improved_convolution(I, K)

    plt.plot(C, 'g')
    plt.plot(I)

    plt.show()


# nal2c()

def g_kernel(range, sig):
    return (1/(((2*math.pi )**(1/2))*sig)) * (np.exp(-((np.power(range, 2))/(2*(sig**2)))))


def nal2d():

    plt.plot(np.arange(-2,3, 1), g_kernel(np.arange(-2,3, 1), 0.5), label='sigma = 0.5')
    plt.plot(np.arange(-3, 4, 1), g_kernel(np.arange(-3, 4, 1), 1), 'orange', label='sigma = 1')
    plt.plot(np.arange(-6, 7, 1), g_kernel(np.arange(-6, 7, 1), 2), 'g', label='sigma = 2')
    plt.plot(np.arange(-9, 10, 1), g_kernel(np.arange(-9, 10, 1), 3), 'r', label='sigma = 3')
    plt.plot(np.arange(-12, 13, 1), g_kernel(np.arange(-12, 13, 1), 4), 'purple', label='sigma = 4')
    plt.legend( loc='upper right')

    plt.show()

# nal2d()

def nal2e():
    I = read_data('signal.txt')


    K1 = g_kernel(np.arange(-6, 7, 1), 2)
    K2 = [0.1, 0.6, 0.4]
    K2 = np.array(K2)

    # print(K1)
    plt.subplot(1,5,1)
    plt.title('Original')
    plt.plot(I)
    plt.subplot(1,5,2)
    plt.title('(I * K1) * K2')
    C1 = improved_convolution(I, K1)
    C2 = improved_convolution(C1, K2)
    plt.plot(C2)
    
    plt.subplot(1,5,3)
    plt.title('(I * K2) * K1')
    C1 = improved_convolution(I, K2)
    C2 = improved_convolution(C1, K1)

    plt.plot(C2)

    plt.subplot(1,5,4)
    plt.title('I *(K1 * K2)')

    C1 = improved_convolution(K1, K2)
    C2 = improved_convolution( I, C1)
    plt.plot(C2)


    plt.show()

# nal2e()


def gaussfilter(I):
    g = g_kernel(np.arange(-4, 5, 1), 1)
    I = cv2.filter2D(I,-1, g)
    I = np.array(I)
    I = I.T
    I = cv2.filter2D(I,-1, g)
    I = np.array(I)
    return I.T



def nal3a():

    I = cv2.cvtColor(cv2.imread('images/lena.png'), cv2.COLOR_BGR2GRAY)
    I = np.array(I)/255

    GI = gaussfilter(I)
    GN = gaussfilter(gauss_noise(I))
    SP = gaussfilter(sp_noise(I))

    plt.set_cmap('gray')
    plt.subplot(2, 3, 1)
    plt.title('Original')
    plt.imshow(I)
    plt.subplot(2, 3, 2)
    plt.title('Gausian Noise')
    plt.imshow(gauss_noise(I))
    plt.subplot(2, 3, 3)
    plt.title('Salt and peper')
    plt.imshow(sp_noise(I))
    plt.subplot(2, 3, 4)
    plt.title('Filtered original')
    plt.imshow(GI)
    plt.subplot(2, 3, 5)
    plt.title('Filtered gausian')
    plt.imshow(GN)
    plt.subplot(2, 3, 6)
    plt.title('Filtered salt and pepper')
    plt.imshow(SP)

    plt.show()
"""
Q: Which noise is better removed using the Gaussian filter?
A: Gausian noise
"""
# nal3a()

def nal3b():
    I = cv2.cvtColor(cv2.imread('images/museum.jpg'), cv2.COLOR_BGR2GRAY)
    I = np.array(I)
    k = [[0, -1, 0],
         [-1, 5, -1],
         [0, -1, 0]]

    
    k = np.array(k)

    plt.set_cmap('gray')
    plt.subplot(1,2,1)
    plt.imshow(I)
    plt.subplot(1,2,2)
    plt.imshow(cv2.filter2D(I, -1, k))

    plt.show()

# nal3b()

def simple_median(I, w):
    L = []
    I = np.array(I)
    I = I.flatten()
    for i in range(len(I) - w):
         Z = np.array(I[i:i+w])
         Z.sort()
         L.append(Z[w//2])
    return L

def nal3c():
    S = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0.5,0.5,0.5,0.5,0.5,
        0.5,0.5,0.5,0.5,0.5,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 0.5,
        0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,
        0.5, 0,0,0,0,0,0,0,0,0,0,0,0,0 ]
    # S1 = sp_noise(S)

    S = np.array(S)
    res = np.copy(S)
    # print(S.shape)
    res[np.random.rand(S.shape[0]) < 0.1 / 2] = 1
    res[np.random.rand(S.shape[0]) < 0.1 / 2] = 0

    L = simple_median(res, 5)
    plt.subplot(1,4, 1)
    plt.title('Original')
    plt.plot(S)
    plt.subplot(1,4, 2)
    plt.title('Corupted')
    plt.plot(res)
    plt.subplot(1,4, 3)
    plt.title('Gauss')
    plt.plot(gaussfilter(res))
    plt.subplot(1,4, 4)
    plt.title('Median')
    plt.plot(L)
    plt.show()

# nal3c()

def median(I, w):
    I = np.array(I)
    
    L = np.zeros((len(I), len(I[0])))
    for i in range(len(I) - w):
        for j in range(len(I[0])- w):
            Z = np.array(I[i: w + i, j: w+j])
            Z = Z.flatten()
            Z.sort()
            L[i, j] = np.median(Z)
    return L


def nal3d():
    I = cv2.cvtColor(cv2.imread('images/lena.png'), cv2.COLOR_BGR2GRAY)
    
    I = np.array(I)/255
    MG = median(gauss_noise(I), 3)
    MS = median(sp_noise(I), 3)

    plt.set_cmap('gray')
    plt.subplot(2, 3, 1)
    plt.title('Original')
    plt.imshow(I)
    plt.subplot(2, 3, 2)
    plt.title('Gausian Noise')
    plt.imshow(gauss_noise(I))
    plt.subplot(2, 3, 3)
    plt.title('Salt and peper')
    plt.imshow(sp_noise(I))
    # plt.subplot(2, 3, 4)
    # plt.title('Filtered original')
    # plt.imshow()
    plt.subplot(2, 3, 5)
    plt.title('median filtered gausian')
    plt.imshow(MG)
    plt.subplot(2, 3, 6)
    plt.title('Filtered salt and pepper')
    plt.imshow(MS)

    plt.show()

"""
Q: What is the computational complexity of the Gaussian filter operation?
How about the median filter? What does it depend on? Describe the computational
complexity using the O(*) notation (you can assume n log n complexity for sorting).

A: naj bo fotografija oblike n x n pikslov in kernel dolžine k.
    GAUSIAN FILTER: nad 2d signalom je oblike (I * g) * g
    ena konvolucija stane O(n^2 * k * 2) = n^2
    konvoluciji se izvedeta ena za drugo (vmes eno transponiranje) torej 2 * n^2 oz n^2
    časovna zahtevnost je potem predvsem odvisna od velikosti slike

    MEDIANA: za vsak piksel je potrebno vezti podmatriko velikosti k^2 okoli piksla, 
    ga urediti po velikosti in izpisati mediano. 
    zahtevnost je n^2 * k^2 * log(k^2). kar je časovna zahtevnost.

"""

nal1a()
nal1b()
nal1c()
nal1d(8)
nal1e()
nal2b()
nal2c()
nal2d()
nal2e()
nal3a()
nal3b()
nal3c()
nal3d()