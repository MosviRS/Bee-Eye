#OpenCV module
import cv2
#Modulo para leer directorios y rutas de archivos
import os
#OpenCV trabaja con arreglos de numpy
import numpy
#Obtener el nombre de la persona que estamos capturando
import sys
import imutils
def XML_imge(img,objeto,list,nombre,name_archivo,path_archivo):
        height, width, channels = img.shape
        xml=f"<annotation>\n"+f"<folder>{nombre}</folder>\n"
        xml+=f"<filename>{name_archivo}</filename>\n"
        xml+=f"<path>{path_archivo}</path>\n"
        xml+=f"<size>\n"
        xml+=f"<width>{width}</width>\n"
        xml+=f"<height>{height}</height>\n"
        xml+=f"<depth>{channels}</depth>\n"
        xml+=f"</size>\n"
        xml+=f"<object>\n"
        xml+=f"<name>{nombre}</name>\n"
        xml+=f"<bndbox>\n"
        xml+=f"<xmin>{list[0]}</xmin>\n"
        xml+=f"<ymin>{list[2]}</ymin>\n"
        xml+=f"<xmax>{list[1]}</xmax>\n"
        xml+=f"<ymax>{list[3]}</ymax>\n"
        xml+=f"</bndbox>\n"
        xml+=f"</object>\n"
        xml+=f"</annotation>\n"
        
        return xml


nombre = sys.argv[1]

#Directorio donde se encuentra la carpeta con el nombre de la persona
dir_faces = "C:/Users/user/Downloads"
path = os.path.join(dir_faces,nombre)


x1, y1 = 190, 80
x2, y2 = 450, 398
opcion=int(input('imagenes positivas...1 \nimagenes negativas ...2'))
#Si no hay una carpeta con el nombre ingresado entonces se crea
if not os.path.isdir(path):
    os.mkdir(path)
    
path_neg=path+'/n'
path = path+'/p'

if not os.path.isdir(path):
    os.mkdir(path)

if not os.path.isdir(path_neg):
    os.mkdir(path_neg)       
        

path_xml=dir_faces+"/"+nombre
path_xml = os.path.join(path_xml,f"xml_{nombre}")
if not os.path.isdir(path_xml):
    os.mkdir(path_xml)


#cargamos la plantilla e inicializamos la webcam
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(1)

#img_width, img_height = 112, 92

#Ciclo para tomar fotografias
count = 0
while count < 200:
    #leemos un frame y lo guardamos
    rval, img = cap.read()
    imAux = img.copy()
    
    objeto = imAux[y1:y2,x1:x2]
    objeto = imutils.resize(objeto,width=38)

    #height, width, channels = objeto.shape
    #upper_left = (width // 4, height // 4)
    #bottom_right = (width * 3 // 4, height * 3 // 4)
   
    

    #Metemos la foto en el directorio
    if opcion==1:
        imagenfinal=objeto
        pin=sorted([int(n[:n.find('.')]) for n in os.listdir(path)
            if n[0]!='.' ]+[0])[-1] + 1
        archivo='%s/%s.png' % (path, pin)
       
    else: 
        imagenfinal=objeto
        pin=sorted([int(n[:n.find('.')]) for n in os.listdir(path_neg)
           if n[0]!='.' ]+[0])[-1] + 1
        archivo='%s/%s.png' % (path_neg, pin)
    
   
    name_archivo='%s.png' % (pin)
    name_xml=pin
    #xml=XML_imge(img,imagenfinal,[x1,x2,y1,y2],nombre,name_archivo,archivo)
    grayimage = cv2.cvtColor(imagenfinal, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(archivo, imagenfinal)

    #guaradmos el xml
    #file = open(f"{path_xml}/{name_xml}.xml", "w")
    #file.write(str(xml))
    #file.close()

     # draw in the image
    cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
    #cv2.rectangle(img, upper_left, bottom_right, (0, 255, 0), thickness=1)
    
    #Ponemos el nombre en el rectagulo
    cv2.putText(img,'Producto', (180,100), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))    
    #Contador del ciclo
    count += 1

    #Mostramos la imagen
    cv2.imshow('OpenCV Entrenamiento de '+nombre, img)
    cv2.imshow('objeto',objeto)

    #Si se presiona la tecla ESC se cierra el programa
    key = cv2.waitKey(10)
    if key == 27:
        cv2.destroyAllWindows()
        break
