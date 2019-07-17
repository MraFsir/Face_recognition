import tensorflow as tf
import numpy as np
import os
import cv2
from mtcnn.mtcnn import MTCNN
from sklearn import cross_validation
from task4 import *
tf.enable_eager_execution()

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    detector = MTCNN()
    model = CNN()
    model.load_weights(r"F:\python3\renlianshibie\CNNmodel")
    #DIR = r"F:\python3\renlianshibie\faceImageGray"
    #names_dict = name_dict(DIR)
    names_dict = {'0': 'huajinqing',
                  '1': 'liangchunfu',
                  '2': 'lijunyu',
                  '3': 'linjuncheng',
                  '4': 'linweixin',
                  '5': 'liujunhao',
                  '6': 'xuhaolin',
                  '7': 'zenglingqi',
                  '8': 'zhouyuanxiang',
                  '9': 'zhushichao'}

    print("按z退出摄像头")
    while(True):
        ret, frame = cap.read()  # 读取一帧的图像
        z = detector.detect_faces(frame)
        if(z):
            cv2.rectangle(frame, (z[0]['box'][0], z[0]['box'][1]), (
                z[0]['box'][0]+z[0]['box'][2], z[0]['box'][1]+z[0]['box'][3]), (0, 255, 0), 2)
            cropped = frame[z[0]['box'][1]:z[0]['box'][1]+z[0]['box']
                            [3], z[0]['box'][0]:z[0]['box'][0]+z[0]['box'][2]]
            if(z[0]['box'][0] < 0 or z[0]['box'][0]+z[0]['box'][2] < 0):
                continue
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            cropped = cv2.resize(cropped, (128, 128))
            cropped = np.expand_dims(cropped, -1)
            y_pred = model.predict(tf.convert_to_tensor(
                cropped, dtype=tf.float32)).numpy()
            y_pred = softmax(y_pred)
            maximum = y_pred.max()
            if(maximum >= 0.5):
                cv2.putText(frame, names_dict[str(int(tf.argmax(y_pred, axis=-1)))], (z[0]
                                                                                      ['box'][0], z[0]['box'][1]), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1)
            else:
                cv2.putText(frame, 'unknown', (z[0]['box'][0], z[0]['box'][1]),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1)
        # for(x, y, w, h) in faces:
            # cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2) # 用矩形圈出人脸
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('z'):
            break

    cap.release()  # 释放摄像头
    cv2.destroyAllWindows()
