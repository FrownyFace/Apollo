import tensorflow as tf
import numpy as np

from utils import load_data

class Apollo:
    def __init__(self, params):
        #x, y = load_data(params['train_path'])
        self.path = params['train_path']
        self.batch_size = params['batch_size']
        #self.x = [x[i:i+self.batch_size] for i in range(0, len(x), self.batch_size)][::-1]
        #self.y = [y[i:i+self.batch_size] for i in range(0, len(y), self.batch_size)][::-1]
        self.epochs = params['epochs']
        self.lr = params['lr']
        self.gen_layers = params['gen_layers']
        self.dis_layers = params['dis_layers']

    def gen(self, X):
        gen_W = [tf.Variable(tf.random_normal([i, j], stddev=0.2), name=k+'W') for i, j, k in self.gen_layers]
        gen_B = [tf.Variable(tf.random_normal([j], stddev=0.2), name=k+'b') for i, j, k in self.gen_layers]

        gen_l = X
        for W, B in zip(gen_W, gen_B):
            l_out = tf.nn.xw_plus_b(gen_l, W, B)
            gen_l = tf.nn.relu(l_out)
        return gen_l

    #def dis(self, Z):


    def build_model(self, _X, _Y):
        g = self.gen(_X)
        loss = tf.reduce_mean(tf.losses.absolute_difference(labels=_Y, predictions=g))
        opt = tf.train.AdamOptimizer(self.lr).minimize(loss)
        return opt, loss, g

    def train(self):
        _X = tf.placeholder(shape=[1, 19200], dtype=tf.float32, name="X")
        _Y = tf.placeholder(shape=[1, 19200], dtype=tf.float32, name="Y")
        #gen_opt, dis_opt, gen_loss, dis_loss, p_real, p_gen = self.build_model()
        opt, loss, p = self.build_model(_X, _Y)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for e in range(self.epochs + 1):
            for file in load_data(self.path):
                for idx in range(1, len(file)-25, 25):
                    #print(file[idx:idx+25])
                    #break
                    _, cur_loss, pred = sess.run([opt, loss, p], feed_dict={
                        _X: file[idx-1],
                        _Y: file[idx]
                    })
            if e % 20 == 0:
                print(e, cur_loss)


if __name__ == '__main__':

    params = {
        'gen_layers': [[19200, 100, 'gen_1'],
                       [100, 100, 'gen_2'],
                       [100, 19200, 'gen_3']],

        'dis_layers': [[19200, 100, 'dis_1'],
                       [100, 100, 'dis_2'],
                       [100, 19200, 'dis_3']],

        'lr': 0.001,
        'epochs': 100,
        'train_path': './data/jsb_train.pkl',
        'test_path': 'test_o.csv',
        'batch_size': 25,
        'export_base_path': './export_models',

    }
    a = Apollo(params)
    a.train()
