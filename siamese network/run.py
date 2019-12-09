
""" Siamese implementation using Tensorflow with MNIST example.
This siamese network embeds a 28x28 image (a point in 784D) 
into a point in 2D.

By Youngwook Paul Kwon (young at berkeley.edu)
"""

from tensorflow.examples.tutorials.mnist import input_data  # 导入数据
import tensorflow as tf
import numpy as np
import inference
import visualize
import os

# 读取数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
# 创建会话
sess = tf.InteractiveSession()

# 导入siamese网络相关
siamese = inference.siamese()
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(siamese.loss)
saver = tf.train.Saver()
tf.initialize_all_variables().run()

# 是否导入之前训练的模型
new = True
model_ckpt = 'model.ckpt'
if os.path.isfile(model_ckpt):
    input_var = None
    while input_var not in ['yes', 'no']:
        input_var = input("We found model.ckpt file. Do you want to load it [yes/no]?")
    if input_var == 'yes':
        new = False

# 开始训练
if new:
    for step in range(10000):
        batch_1, label_1 = mnist.train.next_batch(128)  # 随机从训练集中抽取 BATCH_SIZE 个样本输入神经网络
        batch_2, label_2 = mnist.train.next_batch(128)  # 获得样本的像素值和标签
        label_y = (label_1 == label_2).astype('float')

        _, loss_v = sess.run([train_step, siamese.loss], feed_dict={
                            siamese.x1: batch_1,
                            siamese.x2: batch_2,
                            siamese.y_: label_y})

        if np.isnan(loss_v):
            print('Model diverged with loss = NaN')
            quit()

        if step % 10 == 0:
            print('step %d: loss %.3f' % (step, loss_v))

        if step % 1000 == 0 and step > 0:
            saver.save(sess, 'model.ckpt')
            embed = siamese.o1.eval({siamese.x1: mnist.test.images})
            embed.tofile('embed.txt')
else:
    saver.restore(sess, 'model.ckpt')

# visualize result
x_test = mnist.test.images.reshape([-1, 28, 28])
visualize.visualize(embed, x_test)
