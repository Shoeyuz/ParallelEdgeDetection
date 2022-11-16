import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import time




def sobelFilter(a):
    start_time = time.perf_counter()

    img = a
    #kernel for the filter sobel
    vertical_filter = [[-1,-2,-1],[0,0,0],[1,2,1]]
    horizontal_filter = [[-1,0,1],[-2,0,2],[-1,0,1]]

    #shape of image
    n,m,d = img.shape

    edges_img = np.zeros_like(img)

    #application of the vertical and horizontal kernel together
    for row in range(1,n-2):
        for col in range(1,m-2):
            local_pixels = img[row-1:row+2,col-1:col+2,0]
            
            vertical_transformed_pixels = vertical_filter*local_pixels
            vertical_score = (vertical_transformed_pixels.sum())/4

            horizontal_transformed_pixels = horizontal_filter*local_pixels
            horizontal_score = (horizontal_transformed_pixels.sum())/4
                       
            edge_score = math.sqrt(vertical_score**2 + horizontal_score**2)
            edges_img[row,col] = [edge_score]*3


    edges_img = edges_img/edges_img.max()
    finish_time = time.perf_counter()
    print(f"Serial filtering finished in {finish_time-start_time} sec")

    plt.imshow(edges_img)
    plt.show()
    

if __name__=='__main__':
    img = cv2.imread('C:\\Users\\micha\\OneDrive\\Documents\\School\\comp5900\\IMG_45111.jpg', cv2.COLOR_BGR2GRAY)
    #this is to ensure that we are smoothing the image and removing the noise from the image
    blur = cv2.GaussianBlur(img,(15,15),0)
    sobelFilter(blur)



