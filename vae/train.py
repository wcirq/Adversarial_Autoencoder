# -*- coding: utf-8 -*-
# @File train.py
# @Time 2021/7/10 下午10:48
# @Author wcirq
# @Software PyCharm
# @Site
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow.keras import Sequential, layers

tf.random.set_seed(22)
np.random.seed(22)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


# 把多张image保存达到一张image里面去。
def save_images(img, name):
    new_im = Image.new('L', (280, 280))

    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = img[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1

    new_im.save(name)


# 定义超参数
h_dim = 20  # 把原来的784维护降低到20维度；
batchsz = 512  # fashion_mnist
lr = 1e-4

# 数据集加载
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.

# we do not need label auto-encoder大家可以理解为无监督学习,标签其实就是本身，和自己对比；
train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(batchsz * 5).batch(batchsz)

test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.batch(batchsz)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# 搭建模型
class AE(keras.Model):
    # 1. 初始化部分
    def __init__(self):
        super(AE, self).__init__()  # 调用父类的函数

        # Encoders编码, 网络
        self.encoder = Sequential([
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(h_dim)

        ])

        # Decoders解码，网络
        self.decoder = Sequential([
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(784)

        ])

    # 2. 前向传播的过程
    def call(self, inputs, training=None):
        # [b, 784] ==> [b, 10]
        h = self.encoder(inputs)
        # [b, 10] ==> [b, 784]
        x_hat = self.decoder(h)

        return x_hat


# 创建模型
model = AE()
model.build(input_shape=(None, 784))  # 注意这里面用小括号还是中括号是有讲究的。建议使用tuple
model.summary()

optimizer = keras.optimizers.Adam(lr=lr)
# optimizer = tf.optimizers.Adam() 都可以；

for epoch in range(1000):

    for step, x in enumerate(train_db):

        # [b, 28, 28] => [b, 784]
        x = tf.reshape(x, [-1, 784]).numpy()

        with tf.GradientTape() as tape:
            x_rec_logits = model(x)

            # 把每个像素点当成一个二分类的问题；
            rec_loss = tf.losses.binary_crossentropy(x, x_rec_logits, from_logits=True)
            # rec_loss = tf.losses.MSE(x, x_rec_logits)
            rec_loss = tf.reduce_mean(rec_loss)

        grads = tape.gradient(rec_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print('epoch: %3d, step:%4d, loss:%9f' % (epoch, step, float(rec_loss)))
