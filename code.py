import cv2
import pandas as pd
import time
from datetime import datetime
#gray1,frame2,check2
first_frame=None
status_list=[None,None]
times=[]
df=pd.DataFrame(columns=["star","finsh"])
video9=cv2.VideoCapture(0)
while True:
    check2,frame2=video9.read()
    status=0
    gray1=cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    gray1=cv2.GaussianBlur(gray1,(21,21),0)
    if first_frame  is None:
        first_frame=gray1
        continue
    delta_frame=cv2.absdiff(first_frame,gray1)
    thresh_delta=cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]
    thresh_delta=cv2.dilate(thresh_delta,None,iterations=0)
    ( cnts,_)=cv2.findContours(thresh_delta.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contour in cnts:
        if cv2.contourArea(contour)<10000:
            continue
        status=1 
        (x,y,w,h)=cv2.boundingRect(contour)
        cv2.rectangle(frame2,(x,y),(x+w,y+h),(0,255,0),3)
    status_list.append(status)
    status_list=status_list[-2:]
    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())
    cv2.imshow('frame',frame2)
    cv2.imshow('Capturing',gray1)
    cv2.imshow('Delta',delta_frame)
    key1=cv2.waitKey(1)
    if key1==ord('q'):
        break
print(status_list)
print(times)
for i in range(0,len(times),2):
    df=df.append({"start":times[i],"finsh":times[i+1]},ignore_index=True)
df.to_csv('times.csv')    
    
video9.release()
cv2.destroyAllWndows()
