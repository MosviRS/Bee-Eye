import tkinter as tk 
from PIL import Image
from PIL import ImageTk
import cv2
import imutils
from matplotlib import pyplot as plt
import numpy as np
import asyncio
from clases.postProcesamiento import postporcesmiento


class Application(tk.Frame):

    def __init__(self, master=None):
        super().__init__(master)
        self._master = master
        self.pack()
        #self.__create_widgets()
     
        self.__principal_interfaz()
        self.__i=0
        self.imagenGlobal=None
        self.objimg=postporcesmiento()
      
     
       
     
    def create_widgets(self):
        self.hi_there = tk.Button(self)
        self.hi_there["text"] = "Empezar"
        self.hi_there["command"] = self.say_hi
        self.hi_there.pack(side="top")

        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self.master.destroy)
        self.quit.pack(side="bottom")

    def deteccion(self):
            majinBooClassif = cv2.CascadeClassifier('C:/Users/user/Downloads/usb/classifier/cascade.xml')
            if self.cap is not None:
                
                ret,img = self.cap.read()
                frame=img.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                toy = majinBooClassif.detectMultiScale(gray,
                scaleFactor = int(self.scale.get()),
                minNeighbors = int(self.neigbors.get()),
                minSize=(self.sW.get(),self.sH.get()))
                for (x,y,w,h) in toy:
                
                    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                    cv2.putText(frame,'object',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)

                self.imagenGlobal=img.copy()
                im = Image.fromarray(frame)
                img = ImageTk.PhotoImage(image=im)
                lblVideo.configure(image=img)
                lblVideo.image = img
                lblVideo.after(10, self.deteccion)
                #cv2.imshow('frame',frame)
                
             


    #__create_widgets=create_widgets
    def dibujar(self,mask,color,ROI,namecolor,frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        _,contornos,_= cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.rectangle(frame,(10-2,10-2),(630+2,300+2),(0,255,255),1)
        for c in contornos:
            area = cv2.contourArea(c)
            if area > 3000:
                M = cv2.moments(c)
                if (M["m00"]==0): M["m00"]=1
                x = int(M["m10"]/M["m00"])
                y = int(M['m01']/M['m00'])
                #nuevoContorno = cv2.convexHull(c)
                #cv2.circle(frame,(x,y),7,(0,255,0),-1)
                cv2.putText(ROI,namecolor,(x+10,y), font,1,(0,0,255),2,cv2.LINE_AA)
                cv2.drawContours(ROI, [c], 0, color, 3)

    def visualizar(self):
    
       
        azulBajo = np.array([100,100,20],np.uint8)
        azulAlto = np.array([110,255,255],np.uint8)
        amarilloBajo = np.array([15,100,20],np.uint8)
        amarilloAlto = np.array([45,255,255],np.uint8)
        redBajo1 = np.array([0,100,20],np.uint8)
        redAlto1 = np.array([5,255,255],np.uint8)
        redBajo2 = np.array([175,100,20],np.uint8)
        redAlto2 = np.array([179,255,255],np.uint8)
        #Valores de Verde parte baja
        greenBajo1=np.array([30,100,20], np.uint8)
        greenAlto1=np.array([65,255,255], np.uint8)

        #Valores de Violeta parte baja
        violetBajo1=np.array([120,100,20], np.uint8)
        violetAlto1=np.array([145,255,255], np.uint8)

        #Valores de Rosa parte baja
        pinkBajo1=np.array([145,100,20], np.uint8)
        pinkAlto1=np.array([170,255,255], np.uint8)
        
        if self.cap is not None:
            ret, img = self.cap.read()
            frame=img.copy()
            ROI = frame[0:307,0:650]
            
            if ret == True:

                frameHSV = cv2.cvtColor(ROI,cv2.COLOR_BGR2HSV)
                maskAzul = cv2.inRange(frameHSV,azulBajo,azulAlto)
                maskAmarillo = cv2.inRange(frameHSV,amarilloBajo,amarilloAlto)
                maskRed1 = cv2.inRange(frameHSV,redBajo1,redAlto1)
                maskRed2 = cv2.inRange(frameHSV,redBajo2,redAlto2)
                maskRed = cv2.add(maskRed1,maskRed2)
                maskGreen=cv2.inRange(frameHSV,greenBajo1,greenAlto1)
                maskPink=cv2.inRange(frameHSV,pinkBajo1,pinkAlto1)
                maskPurple=cv2.inRange(frameHSV,violetBajo1,violetAlto1)

                self.dibujar(maskAzul,(255,0,0),ROI,'azul',frame)
                self.dibujar(maskAmarillo,(0,255,255),ROI,'amarillo',frame)
                self.dibujar(maskPink,(265,76,90),ROI,'rosa',frame)
                self.dibujar(maskGreen,(113,74,35),ROI,'verde',frame)
                self.dibujar(maskPurple,(265,76,90),ROI,'violeta',frame)
                self.dibujar(maskRed,(0,0,255),ROI,'rojo',frame)

                #frame=ha.iniciar_video(ret,frame,ROI)
                frame = imutils.resize(frame, width=700)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.imagenGlobal=img.copy()
                im = Image.fromarray(frame)
                img = ImageTk.PhotoImage(image=im)
                lblVideo.configure(image=img)
                lblVideo.image = img
                lblVideo.after(10, self.visualizar)
                if cv2.waitKey(2)==ord('f'):
                     cv2.imwrite('postImages/Miel'+str(self.__i)+'.jpg',frame)
                     self.__i+=1
            
                    
            else:
       
                lblVideo.image = ""
                self.cap.release()
       
    def takepicture(self):
       #await asyncio.sleep(2)
        img=self.imagenGlobal
        scale_percent = 50 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        #resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        self.objimg.main(img)
        self.__i+=1
    def detener(self):
        self.cap.release()
        ImagenFondo=cv2.imread("Images/imagenvacia.png")
        ImagenFondo = imutils.resize(ImagenFondo, width=700)
        im = Image.fromarray(  ImagenFondo)
        img = ImageTk.PhotoImage(image=im)
        lblVideo.configure(image=img)
        lblVideo.image = img
        self.Button1["state"]=tk.NORMAL
        self.Button3["state"]=tk.DISABLED

    def iniciar_prueba(self):
        self.cap = cv2.VideoCapture(0)
        self.Button1["state"]=tk.DISABLED
        self.Button3["state"]=tk.NORMAL

        
        if self.cap.isOpened() is not None:
            if int(self.opcion.get())== 1:
                self.visualizar()
            elif int(self.opcion.get())== 2:
                self.deteccion()
            else:
                self.visualizar()

    def principal_interfaz(self):
        self.Button1 = tk.Button(self,width=45)
        self.Button1["text"] = "Empezar"
        self.Button1.grid(column=0,row=0,padx=5,pady=5)
        self.Button1["command"] = self.iniciar_prueba

        self.Button2 = tk.Button(self,width=45)
        self.Button2["text"] = "Terminar"
        self.Button2.grid(column=1,row=0,padx=5,pady=5)
        self.Button2["command"] = self.detener
        self.opcion = tk.IntVar() 
        self.scale= tk.IntVar() 
        self.neigbors= tk.IntVar() 
        self.sW= tk.IntVar() 
        self.sH= tk.IntVar() 

        self.Radiobutton1=tk.Radiobutton(self, text="Color       ", variable=self.opcion,
        value=1)
        self.Radiobutton1.grid(column=0,row=1)
        self.Radiobutton2=tk.Radiobutton(self, text="Deteccion", variable=self.opcion,
        value=2)
        self.Radiobutton2.grid(column=1,row=1)   

        self.scale = tk.Scale(self, variable=self.scale,label='Sacale Factor', from_=3, to=10, 
        orient=tk.HORIZONTAL, length=200, showvalue=5,
        tickinterval=2, resolution=0.01)
        self.scale.grid(column=0,row=3,padx=5,pady=5) 
        
        self.Neighbors = tk.Scale(self,variable=self.neigbors, label='minNeighbors', from_=20, to=100, 
        orient=tk.HORIZONTAL, length=200, showvalue=95,
        tickinterval=20, resolution=0.01)
        self.Neighbors.grid(column=1,row=3,padx=5,pady=5) 

        self.sizew = tk.Scale(self, variable=self.sW,label='Wi', from_=1, to=100, 
        orient=tk.HORIZONTAL, length=200, showvalue=5,
        tickinterval=30, resolution=0.01)
        self.sizew.grid(column=0,row=4,padx=5,pady=5) 
        
        self.sizeh = tk.Scale(self,variable=self.sH, label='HE', from_=1, to=100, 
        orient=tk.HORIZONTAL, length=200, showvalue=95,
        tickinterval=30, resolution=0.01)
        self.sizeh.grid(column=1,row=4,padx=5,pady=5) 

        global lblVideo
        lblVideo=tk.Label(self,height =400)
        lblVideo.grid(column=0,row=5,columnspan=2)
        #lblVideo.bind("<Return>", self.on_enter_usuario_entry)
        ImagenFondo=cv2.imread("Images/imagenvacia.png")
        ImagenFondo = imutils.resize(ImagenFondo, width=700)
        im = Image.fromarray(  ImagenFondo)
        img = ImageTk.PhotoImage(image=im)
        lblVideo.configure(image=img)
        lblVideo.image = img

        self.Button3 = tk.Button(self,width=45)
        self.Button3["text"] = "Analizar"
        self.Button3.grid(column=1,row=7,padx=5,pady=5)
        self.Button3["state"]=tk.DISABLED
        self.Button3["command"]=self.takepicture

      


    __principal_interfaz=principal_interfaz
    

   
    def say_hi(self):
        print("hi there, everyone!")
    def nothitn(self):
        pass

root = tk.Tk()
app = Application(master=root)
app.mainloop()