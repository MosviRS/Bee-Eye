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
    def promedio(self):
       pass
    def dibujar(self,frame,colorname,mask,color):
        _,controno, _ =cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        suma=0
        filas=0
        for c in controno:
            area=cv2.contourArea(c)
            if area > 2000:
                   M=cv2.moments(c)
                   if M['m10']==0: M['m00']=1
                   x=int(M['m10']/M['m00'])
                   y=int(M['m01']/M['m00'])
                   cv2.circle(frame,(x,y),7,color,-1)
                   fort=cv2.FONT_HERSHEY_SIMPLEX
                   cv2.putText(frame,colorname,(x+10,y),fort,0.75,color,1,cv2.LINE_AA)
                   nuevoContorno=cv2.convexHull(c)
                   suma=suma+1
                   cv2.drawContours(frame,[nuevoContorno],0,color,3)

        return {colorname:int(suma)}

    def detectioncolor(self,img):
        listOCnteo=[]
        propBajo1=np.array([0,0,0],np.uint8(8))
        propAlto1=np.array([125,255,30],np.uint8(8))
        #rojo
        redBajo1=np.array([0,100,20],np.uint8(8))
        redAlto1=np.array([8,255,255],np.uint8(8))
        redBajo2=np.array([175,100,20],np.uint8(8))
        redAlto2=np.array([179,255,255],np.uint8(8))
        #eucalipto
        eucaBajo1=np.array([30,100,20],np.uint8(8))
        eucaAlto1=np.array([65,255,255],np.uint8(8))
        #polen
        polenBajo1 = np.array([0,100,0],np.uint8)
        polenAlto1 = np.array([140,190,80],np.uint8)
        #shmapoo
        shaBajo1 = np.array([14,190,130],np.uint8(8))
        shaAlto1 = np.array([18,255,255],np.uint8(8))
        #gomitas
        gomiBajo1 = np.array([8,80,80],np.uint8)
        gomiAlto1 = np.array([19,255,255],np.uint8)
        #miel
        mielBajo1=np.array([0,190,60],np.uint8(8))
        mielAlto1=np.array([30,245,190],np.uint8(8))

        redBajo1=np.array([0,100,20],np.uint8(8))
        redAlto1=np.array([8,255,255],np.uint8(8))
        redBajo2=np.array([175,100,20],np.uint8(8))
        redAlto2=np.array([179,255,255],np.uint8(8))
        fort=cv2.FONT_HERSHEY_SIMPLEX
        frame=img
        HSV=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        maskecu1=cv2.inRange(HSV,eucaBajo1,eucaAlto1)
        maskprop1=cv2.inRange(HSV,propBajo1,propAlto1)
        maskpolen1=cv2.inRange(HSV,polenBajo1,polenAlto1)
        maskshmap1=cv2.inRange(HSV,shaBajo1,shaAlto1)
        maskgomi1=cv2.inRange(HSV,gomiBajo1,gomiAlto1)
        maskmiel1=cv2.inRange(HSV,mielBajo1,mielAlto1)
        maskRead1=cv2.inRange(HSV,redBajo1,redAlto1)
        maskRead2=cv2.inRange(HSV,redBajo2,redAlto2)
        maskRed=(cv2.add(maskRead1,maskRead2))
        
        propdic=self.dibujar(frame,'propoleo',maskprop1,(255,0,0))
        eucdic=self.dibujar(frame,'eucalipto',maskecu1,(0,255,255))
        poldic=self.dibujar(frame,'polen', maskpolen1,(0,0,255))
        shadic=self.dibujar(frame,'shampoo',maskshmap1,(0,0,255))
        gomdict=self.dibujar(frame,'gomitas', maskgomi1,(0,0,255))
        miedict=self.dibujar(frame,'miel', maskmiel1,(0,0,255))
    
        listOCnteo=[propdic,eucdic,poldic,shadic,gomdict,miedict]
        return listOCnteo

    def canny(self,img):
        #aplicacionde suavizado para resaltar cointornos
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
        #deteccion por texturas
        conteo=0
        print('Extrayendo features...')
        listaConteo=[]
        # Ciclamos por las imágenes de prueba.
        model=joblib.load('clasTexture2.pkl')
        for img in listaImagenes:
            # Cargamos la imagen y la convertimos a escala de grises.
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Calculamos el feature vector usando la implementación de Haralick presente en Mahotas. Fíjate cómo promediamos
            # los cuatro vectores con el método mean().
            features = mahotas.features.haralick(gray).mean(axis=0)
        
            # Imprimimos la predicción (textura) en la imagen.
            prediction = model.predict(features.reshape(1, -1))[0]
            cv2.putText(img, prediction, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            listaConteo.append(str(prediction).replace('train2\\',''))
            listataProductso=[]
            dicttonary={}
            for i in listaConteo:
                try:
                 if dicttonary[i]!= None:
                    dicttonary[i]=dicttonary[i]+1
                except:
                    dicttonary[i]=1
            for key in  dicttonary:
               listataProductso.append({key:dicttonary[key]})

              

        return listataProductso

    def imageHaarDetection(self,frame):
            listaDrop=[]
            dulces = cv2.CascadeClassifier('../haarCascade/dulces.xml')
            polen = cv2.CascadeClassifier('../haarCascade/polen.xml')
            miel = cv2.CascadeClassifier('../haarCascade/miel.xml')

             
            dulcesd = dulces.detectMultiScale(frame,
            scaleFactor = 6,
            minNeighbors = 80,
            minSize=(60,68))

            polend = polen.detectMultiScale(frame,
            scaleFactor = 6,
            minNeighbors = 80,
            minSize=(60,68))

            mield = miel.detectMultiScale(frame,
            scaleFactor = 6,
            minNeighbors = 80,
            minSize=(60,68))
            for (x,y,w,h) in dulcesd:
      
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                im_drop=frame[y:y+h,x:x+w]
                listaDrop.append(im_drop)
            for (x,y,w,h) in polend:
                
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                im_drop=frame[y:y+h,x:x+w]
                listaDrop.append(im_drop)
            for (x,y,w,h) in mield:
                
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                im_drop=frame[y:y+h,x:x+w]
                listaDrop.append(im_drop)

            return listaDrop
                
    
    def getRoisAutomaticImage(self,image):
        ROIs=[[ 66, 183 ,85 ,166],
              [203, 183 ,71 ,165],
              [303, 181 ,80 ,163],
              [417, 187 ,82 ,157],
              [523, 189 ,92 ,158]]
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
        ROIs=[[10,59,56,57],
        [4,59,56,60],
        [17,65,54,54],
        [15,61,54,54],
        [21,56,55,54]]
        crop_number=0 
        listImg=[]
        for image in listImages:
     
                x1=ROIs[crop_number][0]
                y1=ROIs[crop_number][1]
                x2=ROIs[crop_number][2]
                y2=ROIs[crop_number][3]
                
                #crop roi from original image
                img_crop=image[y1:y1+y2,x1:x1+x2]
                #save cropped image
                listImg.append(img_crop)
                #cv2.imwrite("../Images/"+"crop"+str(crop_number)+".jpg",img_crop)      
                crop_number+=1
        
        return listImg

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
        
    def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):

        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow)/255
            gamma_b = shadow

            buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
        else:
            buf = input_img.copy()

        if contrast != 0:
            f = 131*(contrast + 127)/(127*(131-contrast))
            alpha_c = f
            gamma_c = 127*(1-f)

            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf


  

    def main(self,img):
      listaPsroductosTexture=[]
      listaPsroductosColor=[]
      lsitaImagens=[]
    
      #self.getRoisDetectionImage(img)

      #Deteccion por etxtura
      lsitaImagens=self.getRoisAutomaticImage(img)
      lsitaImagens=self.getRoiTextures(lsitaImagens)

      listaPsroductosTexture=self.textureDEtection(lsitaImagens)
      print(listaPsroductosTexture)
      
      #detceion por colores
      listaPsroductosColor=self.detectioncolor(img)
      print(listaPsroductosColor)

      return listaPsroductosColor


      #imglistROI=self.getRoisAutomaticImage(img)      
      #self.imageHaarDetection(imglistROI)
      #imglistROItextures=self.getRoiTextures(imglistROI)
      #listaPsroductos=self.textureDEtection(imglistROItextures)

      #for i in lsitaImagens:
          #dst=cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
          
         # cv2.imshow(f'frame',i)
         # self.brillo(i)
         # cv2.waitKey(0)
    #  cv2.destroyAllWindows()

     

        
#objimg=postporcesmiento()
#img = cv2.imread('../Images/productos2.jpg')
#scale_percent = 50 # percent of original size
#width = int(img.shape[1] * scale_percent / 100)
#height = int(img.shape[0] * scale_percent / 100)
#dim = (width, height)
# resize image
#resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
#cv2.imshow('frame2',img)
#objimg.main(img)






