import tensorflow as tf
import numpy as np
# import mnist_inference
import os
import tqdm

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


def get_result():
    z_dim = 2
    model_path = "/opt/py-project/Adversarial_Autoencoder/Results/Adversarial_Autoencoder/2021-07-12 19:09:56.496976_2_0.001_200_12_0.9_Adversarial_Autoencoder/Saved_models"
    input_checkpoint = tf.train.latest_checkpoint(model_path)
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    with tf.Session() as sess:
        saver.restore(sess, save_path=input_checkpoint)
        op = sess.graph.get_tensor_by_name('Decoder_1/Sigmoid:0')
        decoder_input = sess.graph.get_tensor_by_name('Decoder_input:0')
        results = np.zeros((1, 784))
        for i in range(num):
            result = sess.run(op, feed_dict={decoder_input: np.random.random((1, z_dim))})
            results = np.vstack((results, result))
        return results[1:]


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

    # final_result = np.random.random((100, 784))
    final_result = get_result()
    visualisation(final_result)


if __name__ == '__main__':
    main()