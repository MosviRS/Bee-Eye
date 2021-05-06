import tkinter as tk 
from PIL import Image
from PIL import ImageTk
import cv2
import imutils
from matplotlib import pyplot as plt
import numpy as np
import asyncio
from clases.postProcesamiento import postporcesmiento
import cv2 as cv2
import fontawesome as fa


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
        self.listaConteoProductos=[]
       
       
      
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
        cv2.rectangle(frame,(2,60),(630+2,300+2),(0,255,255),1)
        for c in contornos:
            area = cv2.contourArea(c)
            if area > 1500:
                M = cv2.moments(c)
                if (M["m00"]==0): M["m00"]=1
                x = int(M["m10"]/M["m00"])
                y = int(M['m01']/M['m00'])
                #nuevoContorno = cv2.convexHull(c)
                #cv2.circle(frame,(x,y),7,(0,255,0),-1)
                cv2.putText(ROI,namecolor,(x+10,y), font,1,(0,0,255),2,cv2.LINE_AA)
                cv2.drawContours(ROI, [c], 0, color, 3)

    def visualizar(self):
    
       
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
        polenBajo1 = np.array([15,120,140],np.uint8)
        polenAlto1 = np.array([255,180,220],np.uint8)
        #shmapoo
        shaBajo1 = np.array([14,190,130],np.uint8)
        shaAlto1 = np.array([18,255,255],np.uint8)
        #gomitas
        gomiBajo1 = np.array([14,0,225],np.uint8)
        gomiAlto1 = np.array([100,150,255],np.uint8)
        #miel
        mielBajo1=np.array([0,190,60],np.uint8(8))
        mielAlto1=np.array([30,220,175],np.uint8(8))

        redBajo1=np.array([0,100,20],np.uint8(8))
        redAlto1=np.array([8,255,255],np.uint8(8))
        redBajo2=np.array([175,100,20],np.uint8(8))
        redAlto2=np.array([179,255,255],np.uint8(8))

      
        
        if self.cap is not None:
            ret, img = self.cap.read()
            frame=img.copy()
            ROI = frame[0:307,0:650]
            
            if ret == True:

                frameHSV = cv2.cvtColor(ROI,cv2.COLOR_BGR2HSV)
                maskprop = cv2.inRange(frameHSV,propBajo1,propAlto1)
                maskeuca = cv2.inRange(frameHSV,eucaBajo1,eucaAlto1)

                maskRed1 = cv2.inRange(frameHSV,redBajo1,redAlto1)
                maskRed2 = cv2.inRange(frameHSV,redBajo2,redAlto2)
                maskRed = cv2.add(maskRed1,maskRed2)

                masksha=cv2.inRange(frameHSV,shaBajo1,shaAlto1)
                maskgomi=cv2.inRange(frameHSV,gomiBajo1,gomiAlto1)
                maskpolen=cv2.inRange(frameHSV,polenBajo1,polenAlto1)
                maskmiel=cv2.inRange(frameHSV,mielBajo1,mielAlto1)

                self.dibujar(maskprop,(255,0,0),ROI,'propoleo',frame)
                self.dibujar(maskeuca,(0,255,255),ROI,'eucalipto',frame)
                self.dibujar(masksha,(265,76,90),ROI,'shampoo',frame)
                self.dibujar(maskgomi,(113,74,35),ROI,'gomitas',frame)
                self.dibujar(maskpolen,(265,76,90),ROI,'polen',frame)
                #self.dibujar(maskRed,(0,0,255),ROI,'rojo',frame)
                self.dibujar(maskmiel,(0,0,255),ROI,'miel',frame)

           
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
        
        self.listaConteoProductos=self.objimg.main(img)
        print(self.listaConteoProductos)
        prop='0'
        euca='0'
        pol='0'
        mie='0'
        gomi='0'
        shp='0'
        for i in self.listaConteoProductos:
            for key in i:
                if str(key)=='propoleo':
                     prop=str(i[key])
                if str(key)=='eucalipto':
                    euca=str(i[key])
                if str(key)=='polen':
                    pol=str(i[key])
                if str(key)=='gomitas':
                    gomi=str(i[key])
                if str(key)=='miel':
                     mie=str(i[key])
                if str(key)=='shampoo':
                     shp=str(i[key])
                print(key+str(i[key])+"resild "+prop)
                
       # lblPropoleo['text']=listaConteoProductos
        lblEuca['text']='Eucalipto : '+str(euca)
        lblPropoleo['text']='Propoleo : '+str(prop)
        lblPolen['text']='Polen : '+str(pol)
        lblGomitas['text']='Gomitas : '+str(gomi)
        lblMiel['text']='Miel : '+str(mie)
        lblShampoo['text']='Shampoo : '+str(shp)
        self.__i+=1
    def detener(self):
        self.cap.release()
        ImagenFondo=cv2.imread("Images/backimage.png")
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
    def panlesProductos(self):
        
        global lblPropoleo
        lblPropoleo=tk.Label(self,height=2,text="Propoleo :",width=25)
        lblPropoleo.config(
             font=("Verdana",17),anchor="nw")
        lblPropoleo.grid(column=3,row=0,columnspan=1)
       
        global lblMiel
        lblMiel=tk.Label(self,height=2,text="Miel :",width=25,anchor="w")
        lblMiel.grid(column=3,row=1,columnspan=1,rowspan=1)
        lblMiel.config(
             font=("Verdana",17))
        global lblShampoo
        lblShampoo=tk.Label(self,height=2,text="Shampoo :",width=25,anchor="w")
        lblShampoo.grid(column=3,row=2,columnspan=1,rowspan=1)
        lblShampoo.config(
             font=("Verdana",17)) 
        global lblPolen
        lblPolen=tk.Label(self,height=2,text="Polen :",width=25,anchor="w")
        lblPolen.grid(column=3,row=3,columnspan=1,rowspan=1)
        lblPolen.config(
             font=("Verdana",17))
        global lblEuca
        lblEuca=tk.Label(self,height=2,text="Eucalipto :",width=25,anchor="w")
        lblEuca.grid(column=3,row=4,columnspan=1,rowspan=1)
        lblEuca.config(
             font=("Verdana",17))
        global lblGomitas
        lblGomitas=tk.Label(self,height=2,text="Gomitas :",width=25,anchor="w")
        lblGomitas.grid(column=3,row=5,columnspan=1,rowspan=1)
        lblGomitas.config(
             font=("Verdana",17))    
  
 
  

    def principal_interfaz(self):
       
        self.Button1 = tk.Button(self,width=45,bg='#59D859',fg='white',activebackground='#C7C8C5')
        self.Button1["text"] = "Empezar"
        self.Button1.grid(column=0,row=0,padx=5,pady=5)
        self.Button1["command"] = self.iniciar_prueba

        self.Button2 = tk.Button(self,width=45,bg='#DEB423',fg='black',activebackground='#C7C8C5')
        self.Button2["text"] = "Terminar"
        self.Button2.grid(column=1,row=0,padx=5,pady=5)
        self.Button2["command"] = self.detener
        self.opcion = tk.IntVar() 
        self.scale= tk.IntVar() 
        self.neigbors= tk.IntVar() 
        self.sW= tk.IntVar() 
        self.sH= tk.IntVar() 
        


        self.Radiobutton1=tk.Radiobutton(self, text="Color       ", variable=self.opcion,
        value=1,height=1)
        self.Radiobutton1.grid(column=0,row=1)
        self.Radiobutton2=tk.Radiobutton(self, text="Deteccion", variable=self.opcion,
        value=2,height=1)
        self.Radiobutton2.grid(column=1,row=1)   

        self.scale = tk.Scale(self, variable=self.scale,label='Sacale Factor', from_=3, to=10, 
        orient=tk.HORIZONTAL, length=200, showvalue=5,
        tickinterval=2, resolution=0.01)
        self.scale.grid(column=0,row=2,padx=5,pady=5) 
        self.scale.set(6)
        
        self.Neighbors = tk.Scale(self,variable=self.neigbors, label='minNeighbors', from_=20, to=100, 
        orient=tk.HORIZONTAL, length=200, showvalue=95,
        tickinterval=20, resolution=0.01)
        self.Neighbors.grid(column=1,row=2,padx=5,pady=5) 
        self.Neighbors.set(95)

        self.sizew = tk.Scale(self, variable=self.sW,label='Wi', from_=1, to=100, 
        orient=tk.HORIZONTAL, length=200, showvalue=5,
        tickinterval=30, resolution=0.01)
        self.sizew.grid(column=0,row=3,padx=5,pady=5) 
        self.sizew.set(60)
        
        self.sizeh = tk.Scale(self,variable=self.sH, label='HE', from_=1, to=100, 
        orient=tk.HORIZONTAL, length=200, showvalue=95,
        tickinterval=30, resolution=0.01)
        self.sizeh.grid(column=1,row=3,padx=5,pady=5)
        self.sizeh.set(68) 

        global lblVideo
        lblVideo=tk.Label(self,height =400)
        lblVideo.grid(column=0,row=4,columnspan=2,rowspan=4)
        #lblVideo.bind("<Return>", self.on_enter_usuario_entry)
        ImagenFondo=cv2.imread("Images/backimage.png")
        ImagenFondo = imutils.resize(ImagenFondo, width=700)
        im = Image.fromarray(ImagenFondo)
        img = ImageTk.PhotoImage(image=im)
        lblVideo.configure(image=img)
        lblVideo.image = img

        self.Button3 = tk.Button(self,width=45,bg='#1187A9',fg='#FEFFFD',activebackground='#C7C8C5')
        self.Button3["text"] = "Analizar"
        self.Button3.grid(column=3,row=7,padx=5,pady=5)
        self.Button3["state"]=tk.DISABLED
        self.Button3["command"]=self.takepicture
        self.panlesProductos()
      


    __principal_interfaz=principal_interfaz
    

   
    def say_hi(self):
        print("hi there, everyone!")
    def nothitn(self):
        pass

root = tk.Tk()
app = Application(master=root)
app.mainloop()