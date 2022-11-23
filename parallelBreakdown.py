import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import time
import multiprocessing as mp
from joblib import Parallel, delayed

def splitUpImage(height,width,image,cores):
    imageChunks = []
    sizeX = width / cores
    sizeY = height / cores 
    endWidth = sizeX
    endHeight = sizeY
    x = 0
    y = 0
    for i in range(1,cores):
        for j in range(0,cores+2):
            imageChunks.append(image[int(x):int(endWidth),int(y):int(endHeight)])
            y = endHeight
            endHeight += sizeY
        y = 0
        endHeight = sizeY
        x = endWidth
        endWidth += sizeX 
    return imageChunks


def sobelFilter(a):
    img = a
    #kernel for the filter sobel
    vertical_filter = [[-1,-2,-1],[0,0,0],[1,2,1]]
    horizontal_filter = [[-1,0,1],[-2,0,2],[-1,0,1]]

    #shape of image
    n,m,d = img.shape
    edges_img = np.zeros((n,m))
    #application of the vertical and horizontal kernel together
    for row in range(1,n-2):
        for col in range(1,m-2):
            local_pixels = img[row-1:row+2,col-1:col+2,0]
            
            vertical_transformed_pixels = vertical_filter*local_pixels
            vertical_score = (vertical_transformed_pixels.sum())/4

            horizontal_transformed_pixels = horizontal_filter*local_pixels
            horizontal_score = (horizontal_transformed_pixels.sum())/4
                       
            edge_score = math.sqrt(vertical_score**2 + horizontal_score**2)
            edges_img[row,col] = edge_score*3
    return edges_img


def combine(chunks,height,width,cores,img):
    sliceWidthSize = width / cores
    sliceHeightSize =  height / cores 
    startX = 0
    endX = sliceWidthSize
    startY = 0
    endY = sliceHeightSize
    pos = 0
     #shape of image
    t = np.zeros((height,width))
    for i in range(1, cores):
        for j in range(0, cores+2):
            t[int(startX):int(endX), int(startY):int(endY)] = chunks[pos]
            pos = pos + 1
            startY = endY
            endY += sliceHeightSize
        startY = 0
        endY = sliceHeightSize
        startX = endX
        endX += sliceWidthSize

    return t

    
   

if __name__=='__main__':
    img = cv2.imread('C:\\Users\\micha\\OneDrive\\Documents\\School\\comp5900\\IMG_45111.jpg', cv2.COLOR_BGR2GRAY)
    #this is to ensure that we are smoothing the image and removing the noise from the image
    blur = cv2.GaussianBlur(img,(15,15),0)

    n,m,d = img.shape #rows columns 

    #first step is to split up the image into multiple chunks equal to our processor amount
    chunks = splitUpImage(n,m,blur,mp.cpu_count())

    res = Parallel(n_jobs=mp.cpu_count())(delayed(sobelFilter)(chunk)for chunk in chunks)

    finalImg = combine(res,n,m,mp.cpu_count(),blur)

    finish_time = time.perf_counter()

    plt.imshow(finalImg)
    plt.show()


