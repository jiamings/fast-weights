from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
from associative_retrieval import read_data
from subprocess import call

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
        self.w4 = tf.Variable(tf.random_uniform([100, elem_num], -np.sqrt(1.0 / elem_num), np.sqrt(1.0 / elem_num)),
                              dtype=tf.float32)
        self.b4 = tf.Variable(tf.zeros([1, elem_num]), dtype=tf.float32)

        self.w = tf.Variable(initial_value=0.05 * np.identity(hidden_num), dtype=tf.float32)
        self.c = tf.Variable(tf.random_uniform([100, hidden_num], -np.sqrt(hidden_num), np.sqrt(hidden_num)),
                             dtype=tf.float32)
        self.g = tf.Variable(tf.ones([1, hidden_num]), dtype=tf.float32)
        self.b = tf.Variable(tf.zeros([1, hidden_num]), dtype=tf.float32)

        batch_size = tf.shape(self.x)[0]

        a = tf.zeros(tf.pack([batch_size, hidden_num, hidden_num]), dtype=tf.float32)
        h = tf.zeros([batch_size, hidden_num], dtype=tf.float32)
        la = []
        for t in range(0, step_num):
            z = tf.nn.relu(tf.matmul(
                tf.nn.relu(tf.matmul(self.x[:, t, :], self.w1) + self.b1),
                self.w2) + self.b2
                           )
            h = tf.nn.relu(
                tf.matmul(h, self.w) + tf.matmul(z, self.c)
            )
            hs = tf.reshape(h, tf.pack([batch_size, 1, hidden_num]))
            hh = hs
            a = tf.add(tf.scalar_mul(self.l, a),
                       tf.scalar_mul(self.e, tf.batch_matmul(tf.transpose(hs, [0, 2, 1]), hs)))
            la.append(tf.reduce_mean(tf.square(a)))
            for s in range(0, 1):
                hs = tf.reshape(tf.matmul(h, self.w), tf.shape(hh)) + \
                     tf.reshape(tf.matmul(z, self.c), tf.shape(hh)) + \
                     tf.batch_matmul(hs, a)
                mu = tf.reduce_mean(hs, reduction_indices=0)
                sig = tf.sqrt(tf.reduce_mean(tf.square(hs - mu), reduction_indices=0))
                hs = tf.nn.relu(tf.div(tf.mul(self.g, (hs - mu)), sig) + self.b)
            h = tf.reshape(hs, tf.pack([batch_size, hidden_num]))
        h = tf.nn.relu(tf.matmul(h, self.w3) + self.b3)
        logits = tf.matmul(h, self.w4) + self.b4
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.y))
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss)
        correct = tf.equal(tf.argmax(logits, dimension=1), tf.argmax(self.y, dimension=1))
        self.acc = tf.reduce_mean(tf.cast(correct, tf.float32))
        self.summary = tf.merge_summary([
            tf.scalar_summary('loss', self.loss),
            tf.scalar_summary('acc', self.acc)
        ])
        self.sess = tf.Session()

    def train(self, save=0, verbose=0):
        call('rm -rf ./summary'.split(' '))
        self.sess.run(tf.initialize_all_variables())
        writer = tf.train.SummaryWriter('./summary')
        batch_size = 100
        start_time = time.time()
        saver = tf.train.Saver(tf.all_variables())
        for epoch in range(0, 500):
            batch_idxs = 600
            for idx in range(0, batch_idxs):
                bx, by = ar_data.train.next_batch(batch_size=batch_size)
                loss, acc, summary, _ = self.sess.run([self.loss, self.acc, self.summary, self.trainer],
                                        feed_dict={self.x: bx, self.y: by, self.l: 0.9, self.e: 0.5})
                writer.add_summary(summary, global_step=epoch * batch_idxs + idx)
                if verbose > 0 and idx % verbose == 0:
                    print('Epoch: [{:4d}] [{:4d}/{:4d}] time: {:.4f}, loss: {:.8f}, acc: {:.2f}'.format(
                        epoch, idx, batch_idxs, time.time() - start_time, loss, acc
                    ))
            if save > 0 and (epoch+1) % save == 0:
                saver.save(self.sess, 'log/model', global_step=epoch)
        saver.save(self.sess, 'log/moodel-final')

    def test(self, val=True):
        batch_idxs = 100
        batch_size = 100
        tot = 0.0
        data = ar_data.val if val else ar_data.test
        name = 'Validation' if val else 'Test'
        for idx in range(0, batch_idxs):
            bx, by = data.next_batch(batch_size=batch_size)
            acc = self.sess.run(self.acc, feed_dict={self.x: bx, self.y: by, self.l: 0.9, self.e: 0.5})
            tot += acc / batch_idxs
        print('{}: {:.4f}'.format(name, tot))

    def load(self, save_path='log/model-final'):
        saver = tf.train.Saver(tf.all_variables())
        saver.restore(self.sess, save_path=save_path)


if __name__ == '__main__':
    c = FastWeightsRecurrentNeuralNetworks(STEP_NUM, ELEM_NUM, 20)
    c.train(verbose=10)





