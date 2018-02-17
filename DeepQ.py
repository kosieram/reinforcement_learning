import tensorflow as tf
import numpy as np
import random

from collections import deque

class deepQ_network:
    def __init__(self, topology):
        #Hyper-parameters.
        self.learning_rate = 0.001
        self.gama = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.batch_size = 32
        
        #Build tensor flow model.
        self.percepts_count = topology[0]
        self.action_count = topology[len(topology) - 1]
        self.build_model(topology)

        #Initialize experience memory cache.
        self.memory = deque(maxlen=50000)

    def build_model(self, topology):
        #Initialize tensor flow session.
        tf.reset_default_graph()
        self.sess = tf.Session()
        
        #Construct input layer.
        self.x = tf.placeholder(tf.float32, [None, self.percepts_count])

        #Construct hidden layers.
        f = self.x
        for i in range(1, len(topology)):
            W = tf.Variable(tf.random_uniform([topology[i - 1], topology[i]]))
            b = tf.Variable(tf.random_uniform([topology[i]]))
            f = tf.matmul(f, W) + b
            if i != (len(topology) - 1):
                f = tf.nn.relu(f)
                
        #Construct ouput layers.
        self.y = f
        self.predict = tf.argmax(self.y, 1)

        #Construct training layer.
        self.nextQ = tf.placeholder(tf.float32, [None, self.action_count])
        loss = tf.reduce_sum(tf.square(self.nextQ - self.y))
        trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.update_model = trainer.minimize(loss)
        
        #Initialize trainable variables.
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def get_action(self, percepts, train=False):
        if train and np.random.rand() < self.epsilon:
            return random.randrange(self.action_count)
        return self.sess.run(self.predict, feed_dict={self.x:[percepts]})[0]

    def record(self, percepts, action, reward, next_percepts, done):
        self.memory.append((percepts, action, reward, next_percepts, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0

        minibatch = random.sample(self.memory, self.batch_size)
        for percepts, action, reward, next_percepts, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gama * np.amax(self.sess.run(self.y, feed_dict={self.x:[next_percepts]})[0])

            targetQ = self.sess.run(self.y, feed_dict={self.x:[percepts]})[0]
            targetQ[action] = target

            self.sess.run(self.update_model, feed_dict={self.x:[percepts], self.nextQ:[targetQ]})
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
