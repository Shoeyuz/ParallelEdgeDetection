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


def sobelFilterY(a):
    img = a
    #kernel for the filter sobel
    vertical_filter = [[-1,-2,-1],[0,0,0],[1,2,1]]
    #shape of image
    n,m,d = img.shape
    vertical_edges_img = np.zeros((n,m))
    for row in range(1,n-2):
        for col in range(1,m-2):
            local_pixels = img[row-1:row+2,col-1:col+2,0]
            transformed_pixels = vertical_filter*local_pixels
            vertical_score = (transformed_pixels.sum() + 4)/8
            vertical_edges_img[row,col] = vertical_score*3
    return(vertical_edges_img)

def sobelFilterX(a):
    img = a
    #kernel for the filter sobel
    horizontal_filter = [[-1,0,1],[-2,0,2],[-1,0,1]]
    #shape of image
    n,m,d = img.shape
    horizontal_edges_img = np.zeros((n,m))
    #application of the vertical and horizontal kernel together
    for row in range(1,n-2):
        for col in range(1,m-2):
            local_pixels = img[row-1:row+2,col-1:col+2,0]
            transformed_pixels = horizontal_filter*local_pixels
            horizontal_score = (transformed_pixels.sum() + 4)/8
            horizontal_edges_img[row,col] = horizontal_score*3
    return horizontal_edges_img



def combine(chunks,height,width,cores,q):
    
    sizeX = width / cores
    sizeY =  height / cores 
    x = 0
    endX = sizeX
    y = 0
    endY = sizeY
    pos = 0
     #shape of image
    finalImage = np.zeros((height,width))
    for i in range(1, cores):
        for j in range(0, cores+2):
            finalImage[int(x):int(endX), int(y):int(endY)] = chunks[pos]
            pos = pos + 1
            y = endY
            endY += sizeY
        y = 0
        endY = sizeY
        x = endX
        endX += sizeX
    q.put(finalImage)

    

    
   

if __name__=='__main__':
    img = cv2.imread('C:\\Users\\micha\\OneDrive\\Documents\\School\\comp5900\\IMG_45111.jpg', cv2.COLOR_BGR2GRAY)
    #this is to ensure that we are smoothing the image and removing the noise from the image
    blur = cv2.GaussianBlur(img,(15,15),0)

    n,m,d = img.shape #rows columns 

    #first step is to split up the image into multiple chunks equal to our processor amount
    start_time = time.perf_counter()
    chunks = splitUpImage(n,m,blur,round(mp.cpu_count()/2)-1)


    resY = Parallel(n_jobs=round(mp.cpu_count()/2)-1)(delayed(sobelFilterX)(chunk)for chunk in chunks)
    resX = Parallel(n_jobs=round(mp.cpu_count()/2)-1)(delayed(sobelFilterY)(chunk)for chunk in chunks)
    ctx = mp.get_context('spawn')
    q = ctx.Queue()  

    finalImgX = ctx.Process(target=combine,args=(resX,n,m,round(mp.cpu_count()/2)-1,q))
    finalImgY = ctx.Process(target=combine,args=(resY,n,m,round(mp.cpu_count()/2)-1,q))
    finalImgX.start()
    finalImgY.start()
    #finalImgX = combine(resX,n,m,round(mp.cpu_count()/2)-1)
    #finalImgY = combine(resY,n,m,round(mp.cpu_count()/2)-1)
    sobelCombined = cv2.bitwise_or(q.get(), q.get())
    finalImgY.join()
    finalImgX.join()
    finish_time = time.perf_counter()
    print(f"Parallel filtering finished in {finish_time-start_time} sec")

    plt.imshow(sobelCombined)
    plt.show()


