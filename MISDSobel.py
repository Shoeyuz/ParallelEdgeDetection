import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import multiprocessing as mp
import time


  
#kernel for the filter sobel
vertical_filter = [[-1,-2,-1],[0,0,0],[1,2,1]]
horizontal_filter = [[-1,0,1],[-2,0,2],[-1,0,1]]
def computationVertical(img,vertical_edges_img,n,m,q):
    for row in range(1,n-2):
        for col in range(1,m-2):
            local_pixels = img[row-1:row+2,col-1:col+2,0]
            transformed_pixels = vertical_filter*local_pixels
            vertical_score = (transformed_pixels.sum() + 4)/8
            vertical_edges_img[row,col] = [vertical_score]*3
    q.put(vertical_edges_img)


def computationHoriztonal(img, horizontal_edges_img,n,m,q):
    for row in range(1,n-2):
        for col in range(1,m-2):
            local_pixels = img[row-1:row+2,col-1:col+2,0]
            transformed_pixels = horizontal_filter*local_pixels
            horizontal_score = (transformed_pixels.sum() + 4)/8
            horizontal_edges_img[row,col] = [horizontal_score]*3
    q.put(horizontal_edges_img)

        

if __name__=='__main__':
    start_time = time.perf_counter()
    img = cv2.imread('C:\\Users\\micha\\OneDrive\\Documents\\School\\comp5900\\IMG_45111.jpg', cv2.COLOR_BGR2GRAY)
    #this is to ensure that we are smoothing the image and removing the noise from the image
    #shape of image
    img = cv2.GaussianBlur(img,(7,7),2)

    n,m,d = img.shape
    edges_img = np.zeros_like(img)
    horizontal_edges_img = np.zeros_like(img)
    vertical_edges_img = np.zeros_like(img)

   
    #simple parallelization. splitting the x kernel and y kernel application
    #max change possible is a factor of 2. even then, this is a very simple approach - for benchamrking future ones
    ctx = mp.get_context('spawn')
    q = ctx.Queue()  
    p1 = ctx.Process(target=computationVertical,args=(img,vertical_edges_img,n,m,q))
    p2 = ctx.Process(target=computationHoriztonal,args=(img,horizontal_edges_img,n,m,q))
    
    p1.start()
    p2.start()
    sobelCombined = cv2.bitwise_or(q.get(), q.get())
    p1.join()
    p2.join()
    finish_time = time.perf_counter()


    plt.imshow(sobelCombined)
    plt.show()
    print(f"Parallel filtering finished in {finish_time-start_time} sec")


    
