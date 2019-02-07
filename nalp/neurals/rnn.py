import tensorflow as tf
from nalp.core.neural import Neural
import nalp.utils.decorators as d

class RNN(Neural):

    def __init__(self):
        super(RNN, self).__init__()
        self.model
        self.loss
        self.optimizer
        self.prediction

    @d.define_scope
    def model(self, hidden_size=24, n_class=12):
        self.W = tf.Variable(tf.random_normal([hidden_size, n_class]))
        
        self.b = tf.Variable(tf.random_normal([n_class]))

        self.cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)

        outputs, states = tf.nn.dynamic_rnn(self.cell, self.x, dtype=tf.float32)

        outputs = tf.transpose(outputs, [1, 0, 2])
        
        outputs = outputs[-1]
        
        return tf.matmul(outputs, self.W) + self.b

    @d.define_scope
    def loss(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.model, labels=self.y)
        loss = tf.reduce_mean(cross_entropy)
        return loss
    
    @d.define_scope
    def optimizer(self):
        loss = self.loss
        optimizer = tf.train.AdamOptimizer(0.001)
        return optimizer.minimize(loss)

    @d.define_scope
    def prediction(self):
        return tf.cast(tf.argmax(self.model, 1), tf.int32)

    def train(self, input_batch, target_batch):
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        for epoch in range(5000):
            _, loss = sess.run([self.optimizer, self.loss], feed_dict={self.x: input_batch, self.y: target_batch})
            if (epoch + 1)%1000 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        saver.save(sess, './model')

    def predict(self, input_batch):
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, './model')
        predict =  sess.run([self.prediction], feed_dict={self.x: input_batch})
        return predict