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
        self.x = tf.placeholder(tf.float32, [None, step_num, elem_num])
        self.y = tf.placeholder(tf.float32, [None, elem_num])
        self.l = tf.placeholder(tf.float32, [])
        self.e = tf.placeholder(tf.float32, [])

        self.w1 = tf.Variable(tf.random_uniform([elem_num, 50], -np.sqrt(0.02), np.sqrt(0.02)), dtype=tf.float32)
        self.b1 = tf.Variable(tf.zeros([1, 50]), dtype=tf.float32)
        self.w2 = tf.Variable(tf.random_uniform([50, 100], -np.sqrt(0.01), np.sqrt(0.01)), dtype=tf.float32)
        self.b2 = tf.Variable(tf.zeros([1, 100]), dtype=tf.float32)
        self.w3 = tf.Variable(tf.random_uniform([hidden_num, 100], -np.sqrt(0.01), np.sqrt(0.01)), dtype=tf.float32)
        self.b3 = tf.Variable(tf.zeros([1, 100]), dtype=tf.float32)
        self.w4 = tf.Variable(tf.random_uniform([100, elem_num], -np.sqrt(1.0/elem_num), np.sqrt(1.0/elem_num)), dtype=tf.float32)
        self.b4 = tf.Variable(tf.zeros([1, elem_num]), dtype=tf.float32)

        self.w = tf.Variable(initial_value = 0.05 * np.identity(hidden_num), dtype=tf.float32)
        self.c = tf.Variable(tf.random_uniform([100, hidden_num], -np.sqrt(hidden_num), np.sqrt(hidden_num)), dtype=tf.float32)
        self.g = tf.Variable(tf.ones([1, hidden_num]), dtype=tf.float32)
        self.b = tf.Variable(tf.zeros([1, hidden_num]), dtype=tf.float32)

        batch_size = tf.shape(self.x)[0]

        a = tf.zeros(tf.pack([batch_size, hidden_num, hidden_num]), dtype=tf.float32)
        h = tf.zeros([batch_size, hidden_num], dtype=tf.float32)

        cell = FastRNNCell(hidden_num)

        for t in range(0, step_num):
            z = tf.nn.relu(tf.nn.relu(tf.matmul(
                tf.nn.relu(tf.matmul(self.x[:, t, :], self.w1) + self.b1),
                self.w2) + self.b2
            ))
            _, (h, a) = cell(z, tf.nn.rnn_cell.LSTMStateTuple(h, a))

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
                print('Epoch: [{:4d}] [{:4d}/{:4d}] time: {:.4f}, loss: {:.8f}, acc: {:.2f}'.format(
                    epoch, idx, batch_idxs, time.time() - start_time, loss, acc
                ))
            if save > 0 and (epoch+1) % save == 0:
                saver.save(self.sess, 'log/model', global_step=epoch)


if __name__ == '__main__':
    c = FastWeightsRecurrentNeuralNetworks(STEP_NUM, ELEM_NUM, 20)
    c.train()
