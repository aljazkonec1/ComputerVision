
def nal3e():
    I = cv2.cvtColor(cv2.imread('images/obama.jpg'), cv2.COLOR_BGR2GRAY)

    I = gaussfilter(I)

    l_kernel = [[0,-1,0],
                [-1,4,-1],
                [0, -1, 0]]
    
    l_kernel = np.array(l_kernel)

    L = cv2.filter2D(I, -1, l_kernel)

    plt.set_cmap('gray')
    plt.imshow(L)
    plt.show()


nal3e()
