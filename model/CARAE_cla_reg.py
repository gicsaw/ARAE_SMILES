import numpy as np
import tensorflow as tf
import numpy as np
import threading
from itertools import compress

class ARAE():
    def __init__(self,
                 vocab_size,
                 batch_size=20,
                 latent_size=100,
                 sample_size=100,
                 classification_task=1,
                 regression_task=1):
        print ('batch size : ', batch_size)
        print ('latent size : ', latent_size)
        print ('sample size : ', sample_size)
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.sample_size = sample_size

        self.classification_task = classification_task
        self.regression_task = regression_task
        self.property_task = regression_task + classification_task

        self._create_network()

    def _create_network(self):
        self.X = tf.placeholder(tf.int32, [self.batch_size, None],name="X")     # input smiles
        self.Y = tf.placeholder(tf.int32, [self.batch_size, None],name="Y")     # reconstructed smiles
        self.S = tf.placeholder(tf.float32, [self.batch_size, self.sample_size],name="S")   # seed
        self.L = tf.placeholder(tf.int32, [self.batch_size],"L")    # actual length of SMILES
        self.N = tf.placeholder(tf.float32, [self.batch_size, self.latent_size],"N")    # randomness on latent vectors
        self.P = tf.placeholder(tf.float32, [self.batch_size, self.property_task],"P")    # properties
        mol_onehot = tf.one_hot(tf.cast(self.X, tf.int32), self.vocab_size)
        mol_onehot = tf.cast(mol_onehot, tf.float32)
        self.prefn = [self.latent_size, self.latent_size, self.property_task]
        self.disfn = [self.latent_size, self.latent_size, 1]
        self.genfn = [self.latent_size, self.latent_size, self.latent_size]


        decoded_rnn_size = [self.latent_size]
        encoded_rnn_size = [self.latent_size]
        with tf.variable_scope('decode'):
            decode_cell=[]
            for i in decoded_rnn_size[:]:
                decode_cell.append(tf.nn.rnn_cell.LSTMCell(i))
            self.decoder = tf.nn.rnn_cell.MultiRNNCell(decode_cell)
        
        with tf.variable_scope('encode'):
            encode_cell=[]
            for i in encoded_rnn_size[:]:
                encode_cell.append(tf.nn.rnn_cell.LSTMCell(i))
            self.encoder = tf.nn.rnn_cell.MultiRNNCell(encode_cell)
        self.initial_state=self.decoder.zero_state(self.batch_size, tf.float32)

        self.weights = {}
        self.biases = {}

        self.weights['softmax'] = tf.get_variable("softmaxw", initializer=tf.contrib.layers.xavier_initializer(),\
                                  shape=[decoded_rnn_size[-1], self.vocab_size]) 
        self.biases['softmax'] =  tf.get_variable("softmaxb", initializer=tf.contrib.layers.xavier_initializer(), shape=[self.vocab_size])
        
        for i in range(len(self.disfn)):
            name = 'disfw'+str(i+1)
            if i==0:
                self.weights[name] =  tf.get_variable(name, initializer=tf.contrib.layers.xavier_initializer(),\
                                  shape=[self.latent_size, self.disfn[i]])
            else : 
                self.weights[name] =  tf.get_variable(name, initializer=tf.contrib.layers.xavier_initializer(),\
                                  shape=[self.disfn[i-1], self.disfn[i]])
            name = 'disfb'+str(i+1)
            self.biases[name] = tf.get_variable(name, initializer=tf.zeros_initializer(), shape=[self.disfn[i]])
        
        for i in range(len(self.prefn)):
            name = 'clyfw'+str(i+1)
            if i==0:
                self.weights[name] =  tf.get_variable(name, initializer=tf.contrib.layers.xavier_initializer(),\
                                  shape=[self.latent_size, self.prefn[i]])
            else : 
                self.weights[name] =  tf.get_variable(name, initializer=tf.contrib.layers.xavier_initializer(),\
                                  shape=[self.prefn[i-1], self.prefn[i]])
            name = 'clyfb'+str(i+1)
            self.biases[name] = tf.get_variable(name, initializer=tf.zeros_initializer(), shape=[self.prefn[i]])
        
        for i in range(len(self.genfn)):
            name = 'genfw'+str(i+1)
            if i==0:
                self.weights[name] =  tf.get_variable(name, initializer=tf.contrib.layers.xavier_initializer(),\
                                  shape=[self.sample_size, self.genfn[i]])
            else : 
                self.weights[name] =  tf.get_variable(name, initializer=tf.contrib.layers.xavier_initializer(),\
                                  shape=[self.genfn[i-1], self.genfn[i]])
            name = 'genfb'+str(i+1)
            self.biases[name] = tf.get_variable(name, initializer=tf.zeros_initializer(), shape=[self.genfn[i]])

        self.mol_encoded0 = self.total_encoder(mol_onehot)

        self.mol_encoded = tf.nn.l2_normalize(self.mol_encoded0, dim=-1)
        self.latent_vector = self.generator(self.S)
        d_real_logits = self.discriminator(self.mol_encoded)
        d_fake_logits = self.discriminator(self.latent_vector, reuse=True)

        predicted_property = self.predictor(self.mol_encoded)
#        regression = tf.slice(predicted_property,[0,0],[-1,self.regression_task])
#        classified = tf.slice(predicted_property,[0,self.regression_task],[-1,self.classification_task])

        classified = tf.slice(predicted_property,[0,0],[-1,self.classification_task])
        regression = tf.slice(predicted_property,[0,self.classification_task],[-1,self.regression_task])

#        classified = predicted_property
        self.classified_logits = tf.nn.sigmoid(classified)

        self.mol_encoded +=self.N

        self.mol_decoded_softmax, mol_decoded_logits = self.total_decoder(self.mol_encoded, mol_onehot, self.P)

        weights = tf.sequence_mask(self.L, tf.shape(self.X)[1])
        weights = tf.cast(weights, tf.int32)
        weights = tf.cast(weights, tf.float32)
        self.reconstr_loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(
            logits=mol_decoded_logits, targets=self.Y, weights=weights))

        self.g_loss = -tf.reduce_mean(d_fake_logits)
        self.en_loss = (tf.reduce_mean(d_real_logits))

        self.d_loss = (-tf.reduce_mean(d_real_logits)+tf.reduce_mean(d_fake_logits))


#        P_reg=tf.slice(self.P,[0,0],[-1,self.regression_task])
#        P_class=tf.slice(self.P,[0,self.regression_task],[-1,self.classification_task])
        P_class=tf.slice(self.P,[0,0],[-1,self.classification_task])
        P_reg=tf.slice(self.P,[0,self.classification_task],[-1,self.regression_task])

        self.en_regression_loss = -tf.sqrt(tf.reduce_mean(tf.square(regression-P_reg)))
        self.regression_loss = tf.sqrt(tf.reduce_mean(tf.square(regression-P_reg)))

        self.en_classified_loss = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=classified,labels=P_class))
        self.classified_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=classified,labels=P_class))

        self.en_property_loss= self.en_classified_loss + self.en_regression_loss
        self.property_loss= self.classified_loss + self.regression_loss
#        self.en_property_loss = self.en_classified_loss
#        self.property_loss = self.classified_loss

        # Loss
        self.lr = tf.Variable(0.0, trainable=False)

        tvars = tf.trainable_variables()
        ae_list = [var for var in tvars if 'decode' in var.name or 'encode' in var.name or 'softmax' in var.name]
        en_list = [var for var in tvars if 'encode' in var.name]
        gen_list = [var for var in tvars if 'gen' in var.name]
        dis_list = [var for var in tvars if 'dis' in var.name]
        pre_list = [var for var in tvars if 'cly' in var.name]

        print (np.sum([np.prod(v.shape) for v in ae_list]))
        print (np.sum([np.prod(v.shape) for v in en_list]))
        print (np.sum([np.prod(v.shape) for v in dis_list]))
        print (np.sum([np.prod(v.shape) for v in gen_list]))
        print (np.sum([np.prod(v.shape) for v in pre_list]))
        print (np.sum([np.prod(v.shape) for v in tvars]))
        name1 = [v.name for v in ae_list] 
        name2 = [v.name for v in en_list] 
        name3 = [v.name for v in dis_list] 
        name4 = [v.name for v in gen_list] 
        name5 = [v.name for v in pre_list] 

        optimizer1 = tf.train.GradientDescentOptimizer(1.0)
        optimizer2 = tf.train.AdamOptimizer(1e-5)
        optimizer3 = tf.train.AdamOptimizer(2e-6)
        optimizer4 = tf.train.AdamOptimizer(1e-5)
        self.opt1 = optimizer1.minimize(self.reconstr_loss, var_list = ae_list)
        self.opt2 = optimizer1.minimize(self.en_loss, var_list = en_list)
#        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        self.opt3 = optimizer2.minimize(self.g_loss, var_list = gen_list)
        self.opt4 = optimizer3.minimize(self.d_loss, var_list = dis_list)
        self.opt5 = optimizer1.minimize(self.en_property_loss, var_list = en_list)
        self.opt6 = optimizer1.minimize(self.property_loss, var_list = pre_list)
        self.clip_dis = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in dis_list]
 
        self.mol_pred = tf.argmax(self.mol_decoded_softmax, axis=2)
        self.sess = tf.Session()

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.saver = tf.train.Saver(max_to_keep=None)
#        tf.train.start_queue_runners(sess=self.sess)
        print ("Network Ready")

    def discriminator(self, Z, reuse=False):
        Y=Z
        for i in range(len(self.disfn)):
            name_w = 'disfw'+str(i+1)
            name_b = 'disfb'+str(i+1)
            Y = tf.nn.xw_plus_b(Y, self.weights[name_w], self.biases[name_b])
            if i!=len(self.disfn)-1:
                Y = tf.maximum(Y, 0.2*Y)
        return Y
    
    def predictor(self, Z, reuse=False):
        Y=Z
        for i in range(len(self.prefn)):
            name_w = 'clyfw'+str(i+1)
            name_b = 'clyfb'+str(i+1)
#            Y = tf.nn.sigmoid(tf.nn.xw_plus_b(Y, self.weights[name_w], self.biases[name_b]))
            Y = tf.nn.xw_plus_b(Y, self.weights[name_w], self.biases[name_b])
            if i!=len(self.prefn)-1:
                Y = tf.nn.relu(Y)
        return Y
    
    def generator(self, Z, reuse=False):
        Y=Z
        for i in range(len(self.genfn)):
            name_w = 'genfw'+str(i+1)
            name_b = 'genfb'+str(i+1)
            Y = tf.nn.xw_plus_b(Y, self.weights[name_w], self.biases[name_b])
            if i<len(self.genfn)-1:
                Y = tf.nn.relu(Y)
        return tf.nn.tanh(Y)

    def start_threads(self, num_threads=5, coord=None):
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=self.feed_data, args=(coord,))
            thread.start()
            threads.append(thread)
        return threads
        
    def feed_data(self, coord):
        index = 0
        while True:
            if not coord.should_stop():
                r = np.random.randint(len(self.molecules), size=self.batch_size)
                x = self.molecules[r]
                z = np.random.normal(0.0, 1.0, [self.batch_size, self.latent_size])
                self.sess.run(self.enqueue_op, feed_dict = {self.x : x, self.z : z}) 

    def flat_encoder(self, X):
        for i in range(len(self.efn)):
            name_w = 'efw'+str(i+1)
            name_b = 'efb'+str(i+1)
            if i==0:
                Y = tf.nn.relu(tf.nn.xw_plus_b(X, self.weights[name_w], self.biases[name_b]))
            if i==len(self.efn)-1:
                Y = tf.nn.xw_plus_b(X, self.weights[name_w], self.biases[name_b])
            else:
                Y = tf.nn.relu(tf.nn.xw_plus_b(Y, self.weights[name_w], self.biases[name_b]))
        return Y

    def flat_decoder(self, X):
        for i in range(len(self.dfn)):
            name_w = 'dfw'+str(i+1)
            name_b = 'dfb'+str(i+1)
            if i==0:
                Y = tf.nn.relu(tf.nn.xw_plus_b(X, self.weights[name_w], self.biases[name_b]))
            else:
                Y = tf.nn.relu(tf.nn.xw_plus_b(Y, self.weights[name_w], self.biases[name_b]))
        return Y

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    def conv2d_transpose(self, x, W, output_shape):
        return tf.nn.conv2d_transpose(x, W, strides=[1, 1, 1, 1], output_shape = output_shape, padding='SAME')

    def conv_encoder(self, X):
        X = tf.expand_dims(X, 1)
        for i in range (len(self.ecfn)):
            name_w = 'ecw' + str(i+1)
            name_b = 'ecb' + str(i+1)
            if i==0:
                Y = tf.nn.relu(self.conv2d(X, self.weights[name_w])+self.biases[name_b])
            else:
                Y = tf.nn.relu(self.conv2d(Y, self.weights[name_w])+self.biases[name_b])
        retval = tf.reshape(Y, [self.batch_size, -1])
        return retval
    
    def conv_decoder(self, X):
        X = tf.expand_dims(X, 1)
        X = tf.expand_dims(X, -1)

        for i in range (len(self.dcfn)):
            name_w = 'dcw' + str(i+1)
            name_b = 'dcb' + str(i+1)
            output_shape = [self.batch_size, 1, self.seq_length, self.dcfn[i]]
            if i==0:

                Y = tf.nn.relu(self.conv2d_transpose(X, self.weights[name_w], output_shape)+self.biases[name_b])
            else:
                Y = tf.nn.relu(self.conv2d_transpose(Y, self.weights[name_w], output_shape)+self.biases[name_b])
        retval = tf.reshape(Y, [self.batch_size, -1])
        return retval

    def total_encoder(self, X):
        X = tf.reshape(X, [self.batch_size, tf.shape(self.X)[1], self.vocab_size])
        Y = self.rnn_encoder(X)
        return Y
    
    def rnn_decoder(self, X):
        Y1, self.decoder_state = tf.nn.dynamic_rnn(self.decoder, X, dtype=tf.float32, scope = 'decode', sequence_length = self.L, initial_state = self.initial_state)
        return Y1
    
    def rnn_encoder(self, X):
        _, state = tf.nn.dynamic_rnn(self.encoder, X, dtype=tf.float32, scope = 'encode', sequence_length = self.L)
        c,h = state[0]
        return h

    def total_decoder(self, Z, X, P):
        seq_length = tf.shape(self.X)[1]
        Y1 = tf.expand_dims(Z,1)
        pattern = tf.stack([1, tf.shape(self.X)[1], 1])
        Y1 = tf.tile(Y1, pattern)
        P = tf.expand_dims(P, 1)
        P = tf.tile(P, pattern)
        Y1 = tf.concat([Y1, X, P], axis=-1)
        Y2 = self.rnn_decoder(Y1)
        Y2 = tf.reshape(Y2, [self.batch_size*seq_length, -1])
        Y3_logits = tf.matmul(Y2, self.weights['softmax'])+self.biases['softmax']
        Y3_logits = tf.reshape(Y3_logits, [self.batch_size, seq_length, -1])
        Y3 = tf.nn.softmax(Y3_logits)
        return Y3, Y3_logits

    def get_output(self):
        return self.opt, self.cost

    def train(self, x, y, l, p, n_iter_gan,dev):
        n = np.random.normal(0.0, dev, [self.batch_size, self.latent_size])
        opt1_ = self.sess.run(self.opt1, feed_dict = {self.X : x, self.Y : y, self.L : l, self.N : n, self.P : p})
        opt6_ = self.sess.run(self.opt6, feed_dict = {self.X : x, self.Y : y, self.L : l, self.N : n, self.P : p})
        opt5_ = self.sess.run(self.opt5, feed_dict = {self.X : x, self.Y : y, self.L : l, self.N : n, self.P : p})

        for j in range(n_iter_gan):
            s = np.random.normal(0.0, 0.25, [self.batch_size, self.sample_size]).clip(-1.0,1.0)
            for i in range(5):
                opt4_ = self.sess.run(self.opt4, feed_dict = {self.X : x, self.Y : y, self.L : l, self.S : s, self.P : p})
                _ = self.sess.run(self.clip_dis)
            opt3_ = self.sess.run(self.opt3, feed_dict = {self.S : s})
#            opt2_ = self.sess.run(self.opt2, feed_dict = {self.X : x, self.Y : y, self.L : l})
        return 

    def test(self, x, y, l, s, n, p):

        Y, cost1, cost5, cost6, mol_encoded = self.sess.run(
        [ self.mol_pred, self.reconstr_loss, self.property_loss, self.property_loss,
        self.mol_encoded],
        feed_dict = {self.X:x, self.Y : y, self.L :l, self.N: n, self.P : p})

        cost2, cost3, cost4, latent_vector = self.sess.run(
        [ self.g_loss, self.d_loss, self.en_loss, self.latent_vector],
        feed_dict = {self.X:x, self.Y : y, self.L :l, self.N: n, self.S : s, self.P : p}) 

        return Y, cost1, cost2, cost3, cost4, cost5, cost6, latent_vector,mol_encoded

    def test2(self, x, l, n, p):
        Y,mol_encoded = self.sess.run([self.mol_pred, self.mol_encoded], 
        feed_dict = {self.X:x, self.L :l, self.N: n, self.P : p})
        return Y, mol_encoded

    def save(self, ckpt_path, global_step):
        self.saver.save(self.sess, ckpt_path, global_step = global_step)
        print("model saved to '%s'" % (ckpt_path))

    def assign_lr(self, learning_rate):
        self.sess.run(tf.assign(self.lr, learning_rate ))
    
    def restore(self, ckpt_path):
        self.saver.restore(self.sess, ckpt_path)

    def generate_molecule(self, x, z, l, p, decoder_state):
        return self.sess.run([self.mol_decoded_softmax, self.decoder_state], feed_dict={ self.X: x, self.mol_encoded : z, self.L : l, self.initial_state : decoder_state, self.P : p}) 

    def generate_latent_vector(self, s):
        return self.sess.run(self.latent_vector, feed_dict={self.S : s}) 

    def get_latent_vector(self, x, l):
        return self.sess.run(self.mol_encoded, feed_dict={self.X : x, self.L : l})

    def get_decoder_state(self):
        return self.sess.run(self.initial_state)

    def recover_sample_vector_df(self, reuse):
        self.real_latent_vector = tf.placeholder(tf.float32, [self.batch_size, self.latent_size],"real_latent")
        with tf.variable_scope("trials", reuse=reuse):
            self.S_trials = tf.get_variable("trials", initializer=tf.random_uniform([self.batch_size, self.sample_size], -1, 1))
            self.recover_latent_vector = self.generator(self.S_trials, reuse=True)
            self.err_recover = tf.sqrt(tf.reduce_mean(tf.square(self.real_latent_vector-self.recover_latent_vector), -1)+1e-10)
            optimizer = tf.train.AdamOptimizer(0.1)
            self.opt9 = optimizer.minimize(self.err_recover, var_list = [self.S_trials])
            self.clipping = [self.S_trials.assign(tf.clip_by_value(self.S_trials, -1.0, 1.0))]

        global_vars = tf.global_variables()
        is_not_initialized = self.sess.run([~(tf.is_variable_initialized(var)) \
                                   for var in global_vars])
        self.not_initialized_vars = list(compress(global_vars, is_not_initialized))


    def recover_sample_vector(self, latent_vector_real):
        if len(self.not_initialized_vars):
            self.sess.run(tf.variables_initializer(self.not_initialized_vars))

        for i in range(1000):
            _, err_= self.sess.run([self.opt9, self.err_recover],feed_dict={self.real_latent_vector:latent_vector_real})
            _ = self.sess.run([self.clipping])
            if i%100==0:
                print("cost_latent_real", np.mean(np.array(err_)))
        retval = self.sess.run(self.S_trials)
        print ('check')
        return retval
