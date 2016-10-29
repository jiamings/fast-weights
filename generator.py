import numpy as np
import random
import cPickle as pickle

num_train = 60000
num_val = 10000
num_test = 10000

step_num = 4
elem_num = 26 + 10 + 1

x_train = np.zeros([num_train, step_num * 2 + 3, elem_num], dtype=np.float32)
x_val = np.zeros([num_val, step_num * 2 + 3, elem_num], dtype=np.float32)
x_test = np.zeros([num_test, step_num * 2 + 3, elem_num], dtype=np.float32)

y_train = np.zeros([num_train, elem_num], dtype=np.float32)
y_val = np.zeros([num_val, elem_num], dtype=np.float32)
y_test = np.zeros([num_test, elem_num], dtype=np.float32)


def get_one_hot(c):
    a = np.zeros([elem_num])
    if ord('a') <= ord(c) <= ord('z'):
        a[ord(c) - ord('a')] = 1
    elif ord('0') <= ord(c) <= ord('9'):
        a[ord(c) - ord('0') + 26] = 1
    else:
        a[-1] = 1
    return a


def generate_one():
    a = np.zeros([step_num * 2 + 3, elem_num])
    d = {}
    st = ''

    for i in range(0, step_num):
        c = random.randint(0, 25)
        while d.has_key(c):
            c = random.randint(0, 25)
        b = random.randint(0, 9)
        d[c] = b
        s, t = chr(c + ord('a')), chr(b + ord('0'))
        st += s + t
        a[i*2] = get_one_hot(s)
        a[i*2+1] = get_one_hot(t)

    s = random.choice(d.keys())
    t = chr(s + ord('a'))
    r = chr(d[s] + ord('0'))
    a[step_num * 2] = get_one_hot('?')
    a[step_num * 2 + 1] = get_one_hot('?')
    a[step_num * 2 + 2] = get_one_hot(t)
    st += '??' + t + r
    e = get_one_hot(r)
    return a, e

if __name__ == '__main__':
    for i in range(0, num_train):
        x_train[i], y_train[i] = generate_one()

    for i in range(0, num_test):
        x_test[i], y_test[i] = generate_one()

    for i in range(0, num_val):
        x_val[i], y_val[i] = generate_one()

    d = {
        'x_train': x_train,
        'x_test': x_test,
        'x_val': x_val,
        'y_train': y_train,
        'y_test': y_test,
        'y_val': y_val
    }
    with open('associative-retrieval.pkl', 'wb') as f:
        pickle.dump(d, f, protocol=2)
