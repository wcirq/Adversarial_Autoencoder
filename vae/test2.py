# -*- coding: utf-8 -*-
# @File test2.py
# @Time 2021/7/10 下午11:34
# @Author wcirq
# @Software PyCharm
# @Site
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow.keras import Sequential, layers
import numpy as np

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
batchsz = 256  # fashion_mnist
lr = 1e-4

# 数据集加载
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.

# we do not need label auto-encoder大家可以理解为无监督学习,标签其实就是本身，和自己对比；
train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(batchsz * 5).batch(batchsz)

test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.batch(batchsz)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 搭建模型
z_dim = 10


class VAE(keras.Model):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoders编码, 网络
        self.fc1 = layers.Dense(128, activation=tf.nn.relu)
        # 小网路1:均值(均值和方差是一一对应的，所以维度相同)
        self.fc2 = layers.Dense(z_dim)  # get mean prediction
        # 小网路2
        self.fc3 = layers.Dense(z_dim)  # get mean prediction

        # Decoders解码，网络
        self.fc4 = layers.Dense(128)
        self.fc5 = layers.Dense(784)

    # encoder传播的过程
    def encoder(self, x):
        h = tf.nn.relu(self.fc1(x))
        # get mean
        mu = self.fc2(h)
        # get variance
        log_var = self.fc3(h)

        return mu, log_var

    # decoder传播的过程
    def decoder(self, z):
        out = tf.nn.relu(self.fc4(z))
        out = self.fc5(out)

        return out

    def reparameterize(self, mu, log_var):
        eps = tf.random.normal(log_var.shape)

        std = tf.exp(log_var)  # 去掉log, 得到方差；
        std = std ** 0.5  # 开根号，得到标准差；

        z = mu + std * eps
        return z

    def call(self, inputs, training=None):
        # [b, 784] => [b, z_dim], [b, z_dim]
        mu, log_var = self.encoder(inputs)
        # reparameterizaion trick：最核心的部分
        z = self.reparameterize(mu, log_var)

        # decoder 进行还原
        x_hat = self.decoder(z)

        # Variational auto-encoder除了前向传播不同之外，还有一个额外的约束；
        # 这个约束使得你的mu, var更接近正太分布；所以我们把mu, log_var返回；
        return x_hat, mu, log_var


model = VAE()
model.build(input_shape=(128, 784))
optimizer = keras.optimizers.Adam(lr=lr)

for epoch in range(100):

    for step, x in enumerate(train_db):

        x = tf.reshape(x, [-1, 784])
        with tf.GradientTape() as tape:
            # shape
            x_hat, mu, log_var = model(x)

            # 把每个像素点当成一个二分类的问题；
            rec_loss = tf.losses.binary_crossentropy(x, x_hat, from_logits=True)
            # rec_loss = tf.losses.MSE(x, x_rec_logits)
            rec_loss = tf.reduce_mean(rec_loss)

            # compute kl divergence (mu, var) ~ N(0, 1): 我们得到的均值方差和正太分布的；
            # 链接参考: https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
            kl_div = -0.5 * (log_var + 1 - mu ** 2 - tf.exp(log_var))
            kl_div = tf.reduce_mean(kl_div) / batchsz  # x.shape[0]

            loss = rec_loss + 1. * kl_div

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print('epoch: %3d, step:%4d, kl_div: %5f, rec_loss:%9f' % (epoch, step, float(kl_div), float(rec_loss)))

    # evaluation 1: 从正太分布直接sample；
    z = tf.random.normal((batchsz, z_dim))  # 从正太分布中sample这个尺寸的
    logits = model.decoder(z)  # 通过这个得到decoder
    x_hat = tf.sigmoid(logits)
    x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() * 255.
    logits = x_hat.astype(np.uint8)  # 标准的图片格式；
    save_images(x_hat, 'vae_images/sampled_epoch%d.png' % epoch)  # 直接sample出的正太分布；

    # evaluation 2: 正常的传播过程；
    x = next(iter(test_db))
    x = tf.reshape(x, [-1, 784])
    x_hat_logits, _, _ = model(x)  # 前向传播返回的还有mu, log_var
    x_hat = tf.sigmoid(x_hat_logits)
    x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() * 255.
    x_hat = x_hat.astype(np.uint8)  # 标准的图片格式；
    # print(x_hat.shape)
    save_images(x_hat, 'vae_images/rec_epoch%d.png' % epoch)
