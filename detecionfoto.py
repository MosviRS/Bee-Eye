import cv2

majinBooClassif = cv2.CascadeClassifier('C:/Users/user/Desktop/mosvi/OneDrive/Documentos/programas visual basic/SIMUALCION/dev/VisionArtifitial/proyecto_miel/haarCascade/dulces.xml')
frame = cv2.imread('Images/productos.png')
scale_percent = 50 # percent of original size
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
dim = (width, height)
        #resize image
frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
while True:
    
 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    toy = majinBooClassif.detectMultiScale(gray,
    scaleFactor = 6,
    minNeighbors = 95,
    minSize=(60,68))
    for (x,y,w,h) in toy:
      
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'object',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
     
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) == 27:
        break
cv2.release()
cv2.destroyAllWindows()