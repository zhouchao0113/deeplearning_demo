# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

# 样本个数
sample_num = 5
# 设置迭代次数
epoch_num = 2
# 设置一个批次中包含样本个数
batch_size = 3
# 计算每一轮epoch中含有的batch个数
batch_total = int(sample_num / batch_size) + 1


# 生成4个数据和标签
def generate_data(sample_num=sample_num):
    labels = np.asarray(range(0, sample_num))
    images = np.random.random([sample_num, 224, 224, 3])
    print('image size {},label size :{}'.format(images.shape, labels.shape))
    return images, labels


def get_batch_data(batch_size=batch_size):
    images, label = generate_data()
    # 数据类型转换为tf.float32
    images = tf.cast(images, tf.float32)
    label = tf.cast(label, tf.int32)

    # 从tensor列表中按顺序或随机抽取一个tensor准备放入文件名称队列
    input_queue = tf.train.slice_input_producer([images, label], num_epochs=epoch_num, shuffle=False)

    # 从文件名称队列中读取文件准备放入文件队列
    image_batch, label_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=2, capacity=64,
                                              allow_smaller_final_batch=False)
    return image_batch, label_batch


image_batch, label_batch = get_batch_data(batch_size=batch_size)

with tf.Session() as sess:
    # 先执行初始化工作
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # 开启一个协调器
    coord = tf.train.Coordinator()
    # 使用start_queue_runners 启动队列填充
    threads = tf.train.start_queue_runners(sess, coord)

    try:
        while not coord.should_stop():
            print('************')
            # 获取每一个batch中batch_size个样本和标签
            image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
            print(image_batch_v.shape, label_batch_v)
    except tf.errors.OutOfRangeError:  # 如果读取到文件队列末尾会抛出此异常
        print("done! now lets kill all the threads……")
    finally:
        # 协调器coord发出所有线程终止信号
        coord.request_stop()
        print('all threads are asked to stop!')
    coord.join(threads)  # 把开启的线程加入主线程，等待threads结束
    print('all threads are stopped!')
