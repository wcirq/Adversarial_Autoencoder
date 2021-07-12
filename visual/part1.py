import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data

# PROJECTOR需要的日志文件名和地址相关参数
LOG_DIR = 'log'
SPRITE_FILE = 'mnist_sprite.png'
META_FIEL = "mnist_meta.tsv"


# 使用给出的MNIST图片列表生成sprite图像
def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    # sprite图像可以理解成是小图片平成的大正方形矩阵，大正方形矩阵中的每一个元素就是原来的小图片。于是这个正方形的边长就是sqrt(n),其中n为小图片的数量。
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    # 使用全1来初始化最终的大图片。
    spriteimage = np.ones((img_h*n_plots, img_w*n_plots))

    for i in range(n_plots):
        for j in range(n_plots):
            # 计算当前图片的编号
            this_filter = i*n_plots + j
            if this_filter < images.shape[0]:
                # 将当前小图片的内容复制到最终的sprite图像
                this_img = images[this_filter]
                spriteimage[i*img_h:(i + 1)*img_h,
                j*img_w:(j + 1)*img_w] = this_img

    return spriteimage


# 加载MNIST数据。这里指定了one_hot=False,于是得到的labels就是一个数字，表示当前图片所表示的数字。
mnist = input_data.read_data_sets("./Data", one_hot=False)

# 生成sprite图像
to_visualise = 1 - np.reshape(mnist.test.images, (-1, 28, 28))
sprite_image = create_sprite_image(to_visualise)

# 将生成的sprite图片放到相应的日志目录下
path_for_mnist_sprites = os.path.join(LOG_DIR, SPRITE_FILE)
plt.imsave(path_for_mnist_sprites, sprite_image, cmap='gray')
plt.imshow(sprite_image, cmap='gray')

# 生成每张图片对应的标签文件并写道相应的日志目录下
path_for_mnist_metadata = os.path.join(LOG_DIR, META_FIEL)
with open(path_for_mnist_metadata, 'w') as f:
    f.write("Index\tLabel\n")
    for index, label in enumerate(mnist.test.labels):
        f.write("%d\t%d\n"%(index, label))