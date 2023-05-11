import cv2
import cv
import numpy as np
import sqlite3
# import micropip
# await micropip.install('open-cv')


def getProfile(id):
    conn=sqlite3.connect("FaceBase.db")
    cmd="SELECT * FROM People WHERE ID="+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile 
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer/trainingData.yml")
id=0
# font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,1,1,0,1)
# font=cv2.FONT_HERSHEY_COMPLEX
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255, 0, 0)
while(True):
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
        id,conf=rec.predict(gray [y:y+h , x:x+w])
        profile=getProfile(id)
        if(profile!=None):
            cv2.putText(img,str(profile[1]),(x,y+h+30),fontFace,fontScale,fontColor)
            cv2.putText(img,str(profile[2]),(x,y+h+60),fontFace,fontScale,fontColor)
            cv2.putText(img,str(profile[3]),(x,y+h+90),fontFace,fontScale,fontColor)
            cv2.putText(img,str(profile[4]),(x,y+h+120),fontFace,fontScale,fontColor)
        # cv2.putText(img,str(id),(x,y+h),font,2,(255))
        # cv2.cv.putText(cv2.cv.fromarray(img),str(id),(x,y+h),font,(255))
    cv2.imshow("Face",img);
    if(cv2.waitKey(1)==ord('q')):
        break;
cam.release()
cv2.destroyAllWindows()

"""if(id==1):
            id="nishit"
        elif(id==2):
            id="hanuman ji"""
# cv2.putText(img,str(id),(x,y+h),fontFace,fontScale,fontColor)            