#name：当前拍摄人脸的名字
import cv2
import os

capture = cv2.VideoCapture(0)

name=input("名字：")
if not os.path.exists("F:/python3/renlianshibie/facelmages/"+name):
    os.makedirs("F:/python3/renlianshibie/facelmages/"+name)
i=0
while(i<5):
    print(i)
    ret, frame = capture.read()
    cv2.imshow('photo', frame)
    cv2.imwrite("F:/python3/renlianshibie//facelmages/"+name+"/"+str(i)+".jpg", frame)
    i+=1
    cv2.waitKey(20)

capture.release()
cv2.destroyAllWindows()