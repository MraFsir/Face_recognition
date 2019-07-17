import tensorflow as tf
from sklearn import cross_validation
import numpy as np
import os
tf.enable_eager_execution()

class CNN(tf.keras.Model): 
    def __init__(self): 
        super().__init__() 
        self.conv1 = tf.keras.layers.Conv2D( filters=16, # 卷积核数目 
                                            kernel_size=[5, 5], # 感受野大小 
                                            padding="same", # padding 策略 
                                            activation=tf.nn.relu # 激活函数 
                                           ) 
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D( filters=16, # 卷积核数目 
                                            kernel_size=[5, 5], # 感受野大小 
                                            padding="same", # padding 策略 
                                            activation=tf.nn.relu # 激活函数 
                                           )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Reshape(target_shape=(32 * 32 * 16,))
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)
    def call(self, inputs):
        inputs = tf.reshape(inputs, [-1, 128, 128, 1])
        outputs = self.conv1(inputs) # [batch_size, 128, 128, 16] 
        outputs = self.pool1(outputs) # [batch_size, 64, 64, 16] 
        outputs = self.conv2(outputs) # [batch_size, 64, 64, 16]
        outputs = self.pool2(outputs) # [batch_size, 32, 32, 16]
        outputs = self.flatten(outputs) # [batch_size, 32 * 32 * 16] 
        outputs = self.dense1(outputs) # [batch_size, 1024] 
        outputs = self.dense2(outputs) # [batch_size, 10] 
        return outputs
    def predict(self, inputs): 
        logits = self(inputs)
        return logits
    def predict1(self, inputs): 
        logits = self(inputs)
        return tf.argmax(logits, axis=-1)

def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

#加载数据分为测试集训练集
def load_data(data_path,label_path,num_sum):
    data = np.load( data_path )
    label = np.load(label_path)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(data, label, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test

#获取数据
def get_batch(data_X,data_Y,batch_size):
    index = np.random.randint(0, np.shape(data_X)[0], batch_size)
    return data_X[index, :], data_Y[index]

#数据总数
def photo_sum(path):
    DIR = path
    i = 0
    for h_name in os.listdir(DIR):
        path = os.path.join(DIR, h_name)
        if os.path.isdir(path):
            for p_name in os.listdir(path):
                i = i+1
    return i

#改变标签为数字
def label_change(path,labels):
    names_dict = name_dict(path)
    z = np.zeros_like(labels,dtype=np.int32)
    for j in names_dict:
        z[labels == names_dict[j]] = int(j)
    return z

#建立标签名子与数字的字典
def name_dict(path):
    names_dict = {}
    i = 0
    for h_name in os.listdir(path):
        names_dict[str(i)] = h_name
        i+=1
    return names_dict

if __name__ == '__main__':
    num_batches = 100
    batch_size = 100
    learning_rate = 0.001
    DIR = r"F:\python3\renlianshibie\faceImageGray"
    num_sum = photo_sum(DIR)
    data_path =  r"F:\python3\renlianshibie\data.npy"
    labels_path =  r"F:\python3\renlianshibie\labels.npy"
    X_train, X_test, y_train, y_test = load_data(data_path,labels_path,num_sum)
    y_train = label_change(DIR,y_train)
    y_test = label_change(DIR,y_test)
    
    model = CNN()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    for batch_index in range(num_batches): 
        X, y = get_batch(X_train,y_train,batch_size) #huodeshuju
        with tf.GradientTape() as tape:
            y_logit_pred = model(tf.convert_to_tensor(X,dtype=tf.float32))
            loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.convert_to_tensor(y), logits=y_logit_pred)
            print("batch %d: loss %f" % (batch_index, loss.numpy()))
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    X, y = get_batch(X_train,y_train,batch_size)
    y_pred = model.predict1(tf.convert_to_tensor(X,dtype=tf.float32)).numpy()
    print("train accuracy: %f" % (sum(y_pred == y) / batch_size))

    X1, y1 = get_batch(X_test,y_test,200)
    y_pred = model.predict1(tf.convert_to_tensor(X1,dtype=tf.float32)).numpy() 
    print("test accuracy: %f" % (sum(y_pred == y1) / 200))
    
    model.save_weights("CNNmodel")