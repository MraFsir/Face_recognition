from mtcnn.mtcnn import MTCNN
import cv2
import os
DIR = r"F:\python3\renlianshibie\facelmages"
for h_name in os.listdir(DIR):
    if not os.path.exists(r"F:/python3/renlianshibie/faceImageGray/"+h_name):
        os.makedirs(r"F:/python3/renlianshibie/faceImageGray/"+h_name)
    path = os.path.join(DIR, h_name)
    if os.path.isdir(path):
        for p_name in os.listdir(path):
            img = cv2.imread(path+'\\'+p_name)
            detector = MTCNN()
            z = detector.detect_faces(img)
            if(z):
                cv2.rectangle(img, (z[0]['box'][0], z[0]['box'][1]), (z[0]['box'][0]+z[0]['box'][2], z[0]['box'][1]+z[0]['box'][3]), (0, 255, 0), 2)
                cropped = img[ z[0]['box'][1]:z[0]['box'][1]+z[0]['box'][3], z[0]['box'][0]:z[0]['box'][0]+z[0]['box'][2]]
                cropped = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY) 
                cv2.imwrite(r"F:/python3/renlianshibie/faceImageGray/"+h_name+'//'+p_name, cropped)
            else:
                pass