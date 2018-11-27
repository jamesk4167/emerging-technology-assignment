import numpy as np 
import cv2


#import the image
img = cv2.imread("5.jpg");

#Convert the image to grey scale(needs to be done as Mnist images are in greyscale)
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Check to see if the image has been converted
print(gray_image.shape)



