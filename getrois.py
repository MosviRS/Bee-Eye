import cv2
import numpy as np
import glob
import mahotas  # Contiene la implementación de texturas de Haralick que necesitamos.
from sklearn.svm import LinearSVC
import joblib  # Modelo que entrenaremos para clasificar las texturas.
import numpy as numpy
ROIs = cv2.selectROIs("Select Rois",image)
        #print rectangle points of selected roi
print(ROIs)
#Crop selected roi ffrom raw image
#counter to save image with different name
crop_number=0 
lista=[]
        #loop over every bounding box save in array "ROIs"
image=cv2.imgread("")    
for rect in ROIs:
    x1=rect[0]
    y1=rect[1]
    x2=rect[2]
    y2=rect[3]
            
    #crop roi from original image
    img_crop=image[y1:y1+y2,x1:x1+x2]
    cv2.imshow(img_crop)
    #show cropped image              
    crop_number+=1
    cv2.waitKey(0)
cv2.destroyAllWindows()