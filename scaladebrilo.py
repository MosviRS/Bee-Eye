import cv2 as cv2
import numpy as np

mImage = cv2.imread('Images/productos.png')

hsvImg = cv2.cvtColor(mImage,cv2.COLOR_BGR2HSV)

value = 40

vValue = hsvImg[...,2]
hsvImg[...,2] = np.where((255-vValue)<value,255,vValue+value)

HSV=cv2.cvtColor(hsvImg,cv2.COLOR_HSV2BGR)
cv2.imshow('hsv',HSV)


mImage = cv2.imread('Images/productos.png')
hsvImg = cv2.cvtColor(mImage,cv2.COLOR_BGR2HSV)
# decreasing the V channel by a factor from the original
hsvImg[...,2] = hsvImg[...,2]*0.6
HSV=cv2.cvtColor(hsvImg,cv2.COLOR_HSV2BGR)
cv2.imshow('hsvbajo',HSV)


cv2.waitKey(0)
cv2.destroyAllWindows()