import tensorflow as tf
import numpy as np
# import mnist_inference
import os
import tqdm
import matplotlib.pyplot as plt
import cv2
# 加载用于生成PROJECTOR日志的帮助函数
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99

LOG_DIR = 'log'
SPRITE_FILE = 'mnist_sprite.png'
META_FIEL = "mnist_meta.tsv"
TENSOR_NAME = "FINAL_LOGITS"
num = 500


def imshow(frame, name="window", delay=0):
    cv2.imshow(name, frame)
    cv2.waitKey(delay=delay)


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


def build_meta_sprite():
    # 加载MNIST数据。这里指定了one_hot=False,于是得到的labels就是一个数字，表示当前图片所表示的数字。
    mnist = input_data.read_data_sets("./Data", one_hot=False)

    # 生成sprite图像
    to_visualise = 1 - np.reshape(mnist.test.images[:num], (-1, 28, 28))
    sprite_image = create_sprite_image(to_visualise)

    # 将生成的sprite图片放到相应的日志目录下
    path_for_mnist_sprites = os.path.join(LOG_DIR, SPRITE_FILE)
    plt.imsave(path_for_mnist_sprites, sprite_image, cmap='gray')
    plt.imshow(sprite_image, cmap='gray')

    # 生成每张图片对应的标签文件并写道相应的日志目录下
    path_for_mnist_metadata = os.path.join(LOG_DIR, META_FIEL)
    with open(path_for_mnist_metadata, 'w') as f:
        # f.write("Index\tLabel\n")
        # for index, label in enumerate(mnist.test.labels[:100]):
        #     f.write("%d\t%d\n"%(index, label))

        for index, label in enumerate(mnist.test.labels[:num]):
            f.write("%d\n" % (label,))


def get_result():
    mnist = input_data.read_data_sets("./Data", one_hot=False)
    z_dim = 3
    model_path = "/opt/py-project/Adversarial_Autoencoder/Results/Adversarial_Autoencoder/2021-07-13 09:13:38.932779_3_0.001_500_100_0.9_Adversarial_Autoencoder/Saved_models"
    input_checkpoint = tf.train.latest_checkpoint(model_path)
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    with tf.Session() as sess:
        saver.restore(sess, save_path=input_checkpoint)
        op = sess.graph.get_tensor_by_name('Decoder_1/Sigmoid:0')
        decoder_input = sess.graph.get_tensor_by_name('Decoder_input:0')
        x_input = sess.graph.get_tensor_by_name('Input:0')
        encoder_output = sess.graph.get_tensor_by_name('Encoder/e_latent_variable/matmul_1:0')
        decoder_output = sess.graph.get_tensor_by_name('Decoder/Sigmoid:0')
        result = sess.run(encoder_output, feed_dict={x_input: mnist.test.images[:num]})
        return result


# 生成可视化最终输出层向量所需要的日志文件
def visualisation(final_result):
    # 使用一个新的变量来保存最终输出层向量的结果，因为embedding是通过Tensorflow中变量完成的，所以PROJECTOR可视化的都是TensorFlow中的变哇。
    # 所以这里需要新定义一个变量来保存输出层向量的取值
    y = tf.Variable(final_result, name=TENSOR_NAME)

    # 生成会话，初始化新声明的变量并将需要的日志信息写入文件。
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    # summary_writer = tf.summary.FileWriter(LOG_DIR)

    # 通过project.ProjectorConfig类来帮助生成日志文件
    config = projector.ProjectorConfig()
    # 增加一个需要可视化的bedding结果
    embedding = config.embeddings.add()
    # 指定这个embedding结果所对应的Tensorflow变量名称
    embedding.tensor_name = y.name

    # Specify where you find the metadata
    # 指定embedding结果所对应的原始数据信息。比如这里指定的就是每一张MNIST测试图片对应的真实类别。在单词向量中可以是单词ID对应的单词。
    # 这个文件是可选的，如果没有指定那么向量就没有标签。
    embedding.metadata_path = META_FIEL

    # Specify where you find the sprite (we will create this later)
    # 指定sprite 图像。这个也是可选的，如果没有提供sprite 图像，那么可视化的结果
    # 每一个点就是一个小困点，而不是具体的图片。
    embedding.sprite.image_path = SPRITE_FILE
    # 在提供sprite图像时，通过single_image_dim可以指定单张图片的大小。
    # 这将用于从sprite图像中截取正确的原始图片。
    embedding.sprite.single_image_dim.extend([28, 28])

    merged = tf.summary.merge_all()  # 将图形、训练过程等数据合并在一起
    summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)  # 将训练日志写入到logs文件夹下

    # Say that you want to visualise the embeddings
    # 将PROJECTOR所需要的内容写入日志文件。
    projector.visualize_embeddings(summary_writer, config)

    saver = tf.train.Saver()
    saver.save(sess, os.path.join(LOG_DIR, "model"), TRAINING_STEPS)

    summary_writer.close()


# 主函数先调用模型训练的过程，再使用训练好的模型来处理MNIST测试数据，
# 最后将得到的输出层矩阵输出到PROJECTOR需要的日志文件中。
def main(argv=None):
    build_meta_sprite()
    final_result = get_result()
    visualisation(final_result)


if __name__ == '__main__':
    main()