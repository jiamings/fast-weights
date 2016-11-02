from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
from supercell import FastRNNCell
from associative_retrieval import read_data

ar_data = read_data()

STEP_NUM = 11
ELEM_NUM = 26 + 10 + 1

class FastWeightsRecurrentNeuralNetworks(object):
    def __init__(self, step_num, elem_num, hidden_num):
        self.x = tf.placeholder(tf.float32, [100, step_num, elem_num])
        self.y = tf.placeholder(tf.float32, [100, elem_num])
        self.l = tf.placeholder(tf.float32, [])
        self.e = tf.placeholder(tf.float32, [])
        self.w3 = tf.Variable(tf.random_uniform([hidden_num, 100], -np.sqrt(0.01), np.sqrt(0.01)), dtype=tf.float32)
        self.b3 = tf.Variable(tf.zeros([1, 100]), dtype=tf.float32)
        self.w4 = tf.Variable(tf.random_uniform([100, elem_num], -np.sqrt(1.0/elem_num), np.sqrt(1.0/elem_num)), dtype=tf.float32)
        self.b4 = tf.Variable(tf.zeros([1, elem_num]), dtype=tf.float32)

        batch_size = tf.shape(self.x)[0]

        cell = FastRNNCell(hidden_num)
        state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)

        for t in range(0, step_num):
            if t > 0:
                tf.get_variable_scope().reuse_variables()
            h, state = cell(self.x[:, t, :], state)

        h = tf.nn.relu(tf.matmul(h, self.w3) + self.b3)
        logits = tf.matmul(h, self.w4) + self.b4
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.y))
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss)
        correct = tf.equal(tf.argmax(logits, dimension=1), tf.argmax(self.y, dimension=1))
        self.acc = tf.reduce_mean(tf.cast(correct, tf.float32))
        self.sess = tf.Session()

    def train(self, save=0):
        self.sess.run(tf.initialize_all_variables())
        batch_size = 100
        start_time = time.time()
        saver = tf.train.Saver(tf.all_variables())
        for epoch in range(0, 500):
            batch_idxs = 600
            for idx in range(0, batch_idxs):
                bx, by = ar_data.train.next_batch(batch_size=batch_size)
                loss, acc, _ = self.sess.run([self.loss, self.acc, self.trainer],
                                        feed_dict={self.x: bx, self.y: by, self.l: 0.9, self.e: 0.5})
                if (idx % 100 == 0):
                    print('Epoch: [{:4d}] [{:4d}/{:4d}] time: {:.2f}, loss: {:.4f}, acc: {:.4f}'.format(
                        epoch, idx, batch_idxs, time.time() - start_time, loss, acc
                    ))
            if save > 0 and (epoch+1) % save == 0:
                saver.save(self.sess, 'log/model', global_step=epoch)


if __name__ == '__main__':
    c = FastWeightsRecurrentNeuralNetworks(STEP_NUM, ELEM_NUM, 20)
    c.train()
