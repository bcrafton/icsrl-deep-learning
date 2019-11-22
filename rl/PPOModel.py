
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=1000)

from bc_utils.init_tensor import init_filters
from bc_utils.init_tensor import init_matrix

def conv(x, f, s, weights, name):
    _, _, _, nfilters = f
    if weights:
        filters = tf.Variable(weights[name]          , dtype=tf.float32)
        bias    = tf.Variable(weights[name + '_bias'], dtype=tf.float32)
        assert (np.shape(filters) == f)
        assert (np.shape(bias)[0] == nfilters)
    else:
        filters = tf.Variable(init_filters(size=f, init='glorot_uniform'), dtype=tf.float32)
        bias = tf.Variable(np.zeros(nfilters), dtype=tf.float32)

    conv = tf.nn.conv2d(x, filters, [1,s,s,1], 'VALID') + bias
    relu = tf.nn.relu(conv)
    return relu

def dense(x, size, weights, name):
    _, output_size = size
    if weights:
        w = tf.Variable(weights[name]           , dtype=tf.float32)
        b  = tf.Variable(weights[name + '_bias'], dtype=tf.float32)
        assert (np.shape(w) == size)
        assert (np.shape(b)[0] == output_size)
    else:
        w = tf.Variable(init_matrix(size=size, init='glorot_uniform'), dtype=tf.float32)
        b = tf.Variable(np.zeros(output_size), dtype=tf.float32)

    dot = tf.matmul(x, w) + b
    return dot

class PPOModel:
    def __init__(self, sess, nbatch, nclass, epsilon, decay_max, lr, eps, weights, train):

        self.sess = sess
        self.nbatch = nbatch
        self.nclass = nclass
        self.epsilon = epsilon
        self.decay_max = decay_max
        self.lr = lr
        self.eps = eps
        self.weights = weights
        self.train_flag = train

        ##############################################

        self.states = tf.placeholder("float", [None, 84, 84, 4])
        self.advantages = tf.placeholder("float", [None])
        self.rewards = tf.placeholder("float", [None]) 
        
        self.old_actions = tf.placeholder("int32", [None])
        self.old_values = tf.placeholder("float", [None]) 
        self.old_nlps = tf.placeholder("float", [None])

        ##############################################

        conv1 = conv(x=self.states, f=[8,8, 4,32], s=4, weights=weights, name='conv1')
        conv2 = conv(x=conv1,       f=[4,4,32,64], s=2, weights=weights, name='conv2')
        conv3 = conv(x=conv2,       f=[3,3,64,64], s=1, weights=weights, name='conv3')
        flat  = tf.reshape(conv3, [-1, 7*7*64])
        fc    = dense(x=flat, size=(7*7*64, 512), weights=weights, name='dense1')

        self.values        = tf.squeeze(dense(x=fc, size=(512, 1), weights=weights, name='values'), axis=-1)
        self.action_logits =            dense(x=fc, size=(512, 4), weights=weights, name='actions')

        self.action_dists = tf.distributions.Categorical(logits=self.action_logits)
        self.pi = self.action_dists
        
        ##############################################
        
        if self.train_flag:
            self.actions = tf.squeeze(self.pi.sample(1), axis=0)
        else:
            self.actions = self.pi.mode()

        self.nlps1 = self.pi.log_prob(self.actions)
        self.nlps2 = self.pi.log_prob(self.old_actions)

        ##############################################

        global_step = tf.train.get_or_create_global_step()
        epsilon_decay = tf.train.polynomial_decay(self.epsilon, global_step, self.decay_max, 0.001)

        ##############################################

        ratio = tf.exp(self.nlps2 - self.old_nlps)
        ratio = tf.clip_by_value(ratio, 0, 10)
        surr1 = self.advantages * ratio
        surr2 = self.advantages * tf.clip_by_value(ratio, 1 - epsilon_decay, 1 + epsilon_decay)
        policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

        entropy_loss = -tf.reduce_mean(self.pi.entropy())

        clipped_value_estimate = self.old_values + tf.clip_by_value(self.values - self.old_values, -epsilon_decay, epsilon_decay)
        value_loss_1 = tf.squared_difference(clipped_value_estimate, self.rewards)
        value_loss_2 = tf.squared_difference(self.values, self.rewards)
        value_loss = 0.5 * tf.reduce_mean(tf.maximum(value_loss_1, value_loss_2))

        ##############################################

        self.loss = policy_loss + 0.01 * entropy_loss + 1. * value_loss
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=self.eps).minimize(self.loss)

        ##############################################

        global_step = tf.train.get_or_create_global_step()
        self.global_step_op = global_step.assign_add(1)
        
        ##############################################

        self.params = tf.trainable_variables()
        
    def save_weights(self, filename):        
        [conv1, conv1_bias, conv2, conv2_bias, conv3, conv3_bias, dense1, dense1_bias, values, values_bias, actions, actions_bias] = self.sess.run(self.params, feed_dict={})
        weights = {}
        weights['conv1']      = conv1
        weights['conv1_bias'] = conv1_bias
        weights['conv2']      = conv2
        weights['conv2_bias'] = conv2_bias
        weights['conv3']      = conv3
        weights['conv3_bias'] = conv3_bias
        weights['dense1']       = dense1
        weights['dense1_bias']  = dense1_bias
        weights['values']       = values
        weights['values_bias']  = values_bias
        weights['actions']      = actions
        weights['actions_bias'] = actions_bias
        np.save(filename, weights)

    def set_weights(self):
        self.sess.run(self.global_step_op, feed_dict={})

    ##############################################

    def predict(self, state):
        action, value, nlp = self.sess.run([self.actions, self.values, self.nlps1], {self.states:[state]})

        action = np.squeeze(action)
        value = np.squeeze(value)
        nlp = np.squeeze(nlp)
        
        return action, value, nlp
        
    ##############################################

    def train(self, states, rewards, advantages, old_actions, old_values, old_nlps):
        feed_dict={self.states:states, self.rewards:rewards, self.advantages:advantages, self.old_actions:old_actions, self.old_values:old_values, self.old_nlps:old_nlps}
        self.sess.run([self.train_op], feed_dict=feed_dict)

    ##############################################
        
        
        
        
