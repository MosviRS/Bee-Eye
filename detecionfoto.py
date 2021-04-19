import cv2

majinBooClassif = cv2.CascadeClassifier('C:/Users/user/Downloads/opencv/build/x64/vc14/bin/data/cascade.xml')
frame = cv2.imread('../../postImages/kang2.jpg')
while True:
    
 
    

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    toy = majinBooClassif.detectMultiScale(gray,
    scaleFactor = 5,
    minNeighbors = 95,
    minSize=(60,68))
    for (x,y,w,h) in toy:
      
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,'object',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)
     
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()