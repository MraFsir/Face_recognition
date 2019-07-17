import cv2
import numpy as np
def to_npy(path):
    list_data = []
    list_label = []
    DIR = path
    for h_name in os.listdir(DIR):
        path = os.path.join(DIR, h_name)
        if os.path.isdir(path):
            for p_name in os.listdir(path):
                img = cv2.imread(path+'\\'+p_name,0)
                img = cv2.resize(img, (128,128))
                img = np.expand_dims(img,-1)
                list_data.append(img)
                list_label.append(h_name)
    data = np.array(list_data)
    print(data.shape)
    labels = np.array(list_label)
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = data[permutation, :, :,:]
    shuffled_labels = labels[permutation]
    np.save("data.npy",shuffled_dataset)
    np.save("labels.npy",shuffled_labels)
if __name__ == '__main__':
    DIR = r"F:\python3\renlianshibie\faceImageGray"
    to_npy(DIR)