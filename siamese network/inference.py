import tensorflow as tf 


class siamese:

    # 搭建模型
    def __init__(self):  # 类实例化时会自动调用
        self.x1 = tf.placeholder(tf.float32, [None, 784])  # self代表类的实例，而非类
        self.x2 = tf.placeholder(tf.float32, [None, 784])

        with tf.variable_scope("siamese") as scope:
            self.o1 = self.network(self.x1)
            scope.reuse_variables()
            self.o2 = self.network(self.x2)

        # 损失函数
        self.y_ = tf.placeholder(tf.float32, [None])
        self.loss = self.loss_with_spring()

    def network(self, x):  # 类的方法必须有一个额外的第一个参数名称, 按照惯例它的名称是 self
        weights = []
        fc1 = self.fc_layer(x, 1024, "fc1")
        ac1 = tf.nn.relu(fc1)
        fc2 = self.fc_layer(ac1, 1024, "fc2")
        ac2 = tf.nn.relu(fc2)
        fc3 = self.fc_layer(ac2, 2, "fc3")
        return fc3

    def fc_layer(self, bottom, n_weight, name):
        assert len(bottom.get_shape()) == 2
        n_prev_weight = bottom.get_shape()[1]
        initer = tf.truncated_normal_initializer(stddev=0.01)
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
        return fc

    def loss_with_spring(self):
        margin = 5.0
        labels_t = self.y_  # 匹配标签 1
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")  # 不匹配标签 0
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)  # 两个网络标签相减
        eucd2 = tf.reduce_sum(eucd2, 1)  # 横向对矩阵求和
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")  # 创建tf常量
        pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
        neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")  # 平均值
        return loss
