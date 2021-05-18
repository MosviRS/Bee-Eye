import cv2 as cv2

dulces = cv2.CascadeClassifier('../haarCascade/dulces.xml')
polen = cv2.CascadeClassifier('../haarCascade/polen.xml')
miel = cv2.CascadeClassifier('../haarCascade/miel.xml')
frame = cv2.imread('Images/polencolor.png')
scale_percent = 50 # percent of original size
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
dim = (width, height)
        #resize image
frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

    
 
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
lsitaImg=[]

for (x,y,w,h) in dulcesd:
      
    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
    cv2.putText(frame,'dicels',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
    im_drop=frame[y:y+h,x:x+w]
    lsitaImg.append(im_drop)
for (x,y,w,h) in polend:
      
    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
    cv2.putText(frame,'poeln',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
    im_drop=frame[y:y+h,x:x+w]
    lsitaImg.append(im_drop)
for (x,y,w,h) in mield:
      
    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
    cv2.putText(frame,'miel',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
    im_drop=frame[y:y+h,x:x+w]
    lsitaImg.append(im_drop)
     
cv2.imshow('frame',frame)
for img in lsitaImg:
 cv2.imshow('img',img)
 cv2.waitKey(0)
        

cv2.waitKey(0)
cv2.release()
cv2.destroyAllWindows()