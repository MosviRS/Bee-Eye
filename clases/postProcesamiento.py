import cv2
import numpy as np
import glob
import mahotas  # Contiene la implementación de texturas de Haralick que necesitamos.
from sklearn.svm import LinearSVC
import joblib  # Modelo que entrenaremos para clasificar las texturas.
import numpy as numpy
class postporcesmiento:

    def __init__(self):
        pass
    def detectionColor(self):
       pass
    def promedio(self):
       pass
    def canny(self,img):

        # Convertimos a escala de grises
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Aplicar suavizado Gaussiano
        gauss = cv2.GaussianBlur(gris, (5,5), 0)
        canny = cv2.Canny(gauss, 50, 150)
 
        cv2.imshow("canny", canny)
        (_,contornos,_) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
        cv2.drawContours(img,contornos,-1,(0,0,255), 2)
        cv2.imshow("contornos", img)

    def segementacion(self,img):
     
        
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            # Eliminación del ruido
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
        
        # Encuentra el área del fondo
        sure_bg = cv2.dilate(opening,kernel,iterations=3)
        
        # Encuentra el área del primer
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
        
        # Encuentra la región desconocida (bordes)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)

        cv2.imshow('binarizada',thresh)
        cv2.imshow('sin ruido',opening)
        cv2.imshow('area de fondo',sure_bg)
        cv2.imshow('area de frente',sure_fg)
        cv2.imshow('distancia',dist_transform)
        cv2.imshow('bordes',unknown)
        cv2.imshow('regiones',img)
       
    def textureDEtection(self,listaImagenes):
       
        print('Extrayendo features...')
        listaPsroductos=[]
        # Ciclamos por las imágenes de prueba.
        model=joblib.load('clasTexture.pkl')
        for img in listaImagenes:
            # Cargamos la imagen y la convertimos a escala de grises.
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Calculamos el feature vector usando la implementación de Haralick presente en Mahotas. Fíjate cómo promediamos
            # los cuatro vectores con el método mean().
            features = mahotas.features.haralick(gray).mean(axis=0)
        
            # Imprimimos la predicción (textura) en la imagen.
            prediction = model.predict(features.reshape(1, -1))[0]
            cv2.putText(img, prediction, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            listaPsroductos.append(prediction)
        return listaPsroductos

    def imageHaarDetection(self,listaImagenes):
            ProductClassif = cv2.CascadeClassifier('C:/Users/user/Downloads/opencv/build/x64/vc14/bin/data/cascade.xml')
            for i in listaImagenes:
                frame = i

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
    def getRoiTextures(self,listImages):
        ROIs=[[0,2,252,470],
        [252,3,164,469],
        [416,3,176,469],
        [592,4,155,468],
        [749,2,242,470],]
        crop_number=0 
        listImages=[]
        for image in listImages:
      
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
            cv2.imwrite("crop"+str(crop_number)+".jpeg",img_crop)
            #show cropped image
                    
            crop_number+=1
        
    def brillo(self,img):
        cols, rows,_ = img.shape
        brightness = numpy.sum(img) / (255 * cols * rows)
        minimum_brightness = 0.66
        alpha = brightness / minimum_brightness
        bright_img = cv2.convertScaleAbs(img, alpha = alpha, beta = 255 * (1 - alpha))
        #cv2.imshow('frame2',img) 
        return img      
    def main(self,img):
      listaPsroductos=[]
      lsitaImagens=[]
      img=self.brillo(img)
      self.getRoisDetectionImage(img)
     
      #imglistROI=self.getRoisAutomaticImage(img)
      
      #self.imageHaarDetection(imglistROI)
      #imglistROItextures=self.getRoiTextures(imglistROI)
      #listaPsroductos=self.textureDEtection(imglistROItextures)

      #for i in imglistROI:
          #dst=cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
          
         # cv2.imshow(f'frame',i)
         # self.brillo(i)
         # cv2.waitKey(0)
         # cv2.destroyAllWindows()

     

        
#objimg=postporcesmiento()
#img = cv2.imread('../Images/productos.png')
#scale_percent = 50 # percent of original size
#width = int(img.shape[1] * scale_percent / 100)
#height = int(img.shape[0] * scale_percent / 100)
#dim = (width, height)
# resize image
#resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
#objimg.main(resized)
#cv2.imshow('frame',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()



