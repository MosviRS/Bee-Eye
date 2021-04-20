import cv2
import numpy as np
class postporcesmiento:

    def __init__(self):
        pass
    def imageHaarDetection():
           ProductClassif = cv2.CascadeClassifier('C:/Users/user/Downloads/opencv/build/x64/vc14/bin/data/cascade.xml')
           frame = cv2.imread('../../postImages/kang2.jpg')

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            toy = ProductClassif.detectMultiScale(gray,
            scaleFactor = 5,
            minNeighbors = 95,
            minSize=(60,68))
            for (x,y,w,h) in toy:
            
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(frame,'object',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
            
            cv2.imshow('frame',frame)
    
    def getRoisAutomaticImage(self,image):
        ROIs=[[0,2,252,470],
        [252,3,164,469],
        [416,3,176,469],
        [592,4,155,468],
        [749,2,242,470],]
        crop_number=0 
        listImages=[]

        for rect in ROIs:
            x1=rect[0]
            y1=rect[1]
            x2=rect[2]
            y2=rect[3]
            
            #crop roi from original image
            img_crop=image[y1:y1+y2,x1:x1+x2]
            #save cropped image
            listImages.append(img_crop)
            #cv2.imwrite("../Images/"+"crop"+str(crop_number)+".jpg",img_crop)      
            crop_number+=1

        return listImages

        

    def getRoisDetectionImage(self,image):
        
        ROIs = cv2.selectROIs("Select Rois",image)
        #print rectangle points of selected roi
        print(ROIs)
        #Crop selected roi ffrom raw image
        #counter to save image with different name
        crop_number=0 
        
        #loop over every bounding box save in array "ROIs"
        for rect in ROIs:
            x1=rect[0]
            y1=rect[1]
            x2=rect[2]
            y2=rect[3]
            
            #crop roi from original image
            img_crop=image[y1:y1+y2,x1:x1+x2]
            
            #show cropped image
            cv2.imshow("crop"+str(crop_number),img_crop)
            
            #save cropped image
            cv2.imwrite("../Images"+"crop"+str(crop_number)+".jpg",img_crop)
                    
            crop_number+=1
            
        
        

objimg=postporcesmiento()
img = cv2.imread('../Images/productos.png')
scale_percent = 50 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
objimg.getRoisAutomaticImage(resized)
cv2.imshow('frame',img)
cv2.waitKey(0)
cv2.destroyAllWindows()



