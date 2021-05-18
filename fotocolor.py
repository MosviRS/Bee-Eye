import cv2 as cv2
import numpy as np




def dibujar(colorname,mask,color):
        _,controno, _ =cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        listOCnteo=[]
        suma=0
        filas=0
        for c in controno:
            area=cv2.contourArea(c)
            if area > 1700:
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
        #maskRedvis=cv2.bitwise_and(frame,frame,mask= maskblue1)
        #cv2.imshow('videoREdvis',maskRedvis)
        #cv2.imshow('videomadkred',maskblue1)
        listOCnteo.append({colorname:int(suma)})
        cv2.imshow('video',frame)
        print(listOCnteo)
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


if __name__ == "__main__":
      
        
        #propoleo
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
        polenBajo1 = np.array([10,10,30],np.uint8)
        polenAlto1 = np.array([140,190,80],np.uint8)
        #shmapoo
        shaBajo1 = np.array([14,190,130],np.uint8)
        shaAlto1 = np.array([19,255,255],np.uint8)
        #gomitas

        gomiBajo1 = np.array([8,80,80],np.uint8)
        gomiAlto1 = np.array([19,255,255],np.uint8)


        #miel
        mielBajo1=np.array([0,190,60],np.uint8(8))
        mielAlto1=np.array([30,220,175],np.uint8(8))

        fort=cv2.FONT_HERSHEY_SIMPLEX
        frame=cv2.imread('Images/pro.png')
        #frame=cv2.convertScaleAbs(frame, alpha=0, beta=(-40))
        #scale_percent = 50 # percent of original size
        #width = int(frame.shape[1] * scale_percent / 100)
       # height = int(frame.shape[0] * scale_percent / 100)
        #dim = (width, height)
        #resize image
       # resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        #frame=resized
      
        frame=apply_brightness_contrast(frame,-40,0)
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
        
        dibujar('propoleo',maskprop1,(255,0,0))
        dibujar('eucalipto',maskecu1,(0,255,255))
        #dibujar('rojo',maskRed,(0,0,255))
        dibujar('polen', maskpolen1,(0,0,255))
        dibujar('shampoo',maskshmap1,(0,0,255))
        dibujar('gomitas', maskgomi1,(0,0,255))
        dibujar('miel', maskmiel1,(0,0,255))
        
        
        cv2.imshow('video',frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
