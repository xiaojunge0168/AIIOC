
from pipeline.NER.utils import *
from pipeline.NER.lstmcrf import lstmcrf_net

import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
num_chars = 128
dim_chars = 32
hidden_size_char = 64
lstm_unit_num = 128
tags_num = get_class_size()

class char_lstmcrf_net(lstmcrf_net):
    def __init__(self, scope_name,):
        print ("initialize char_lstmcrf_net...")
        super(char_lstmcrf_net, self).__init__(scope_name)
    
    def _build_net(self):
        print ("building char_lstm_crf net...")

        self.source = tf.placeholder(tf.int32, shape=([None, None]))
        self.target = tf.placeholder(tf.int32, shape=([None, None]))
        self.sequence_length = tf.placeholder(tf.int32, shape=([None, ]))
        self.char_source = tf.placeholder(tf.int32, shape=([None, None, None]))
        self.word_lengths = tf.placeholder(tf.int32, shape=([None, None]))
        self.batch_size = tf.shape(self.source)[0]

        word_input = tf.nn.embedding_lookup(self.embedding, self.source, name='word_embedding_lookup')
        char_embedding = tf.get_variable('char_embedding', 
                shape=(num_chars, dim_chars),
                dtype=tf.float32,
                initializer=tf.glorot_uniform_initializer(123),
                trainable=True)
        char_input = tf.nn.embedding_lookup(char_embedding, self.char_source, 
                name = 'char_embedding_lookup') # (batch_size, max_len, max_chars, dim_chars)
        with tf.variable_scope('char-lstm'):
            s = tf.shape(char_input)
            char_input = tf.reshape(char_input, shape=[s[0]*s[1], s[2], dim_chars])
            word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])
            
            cell_fw = tf.contrib.rnn.BasicLSTMCell(hidden_size_char)
            cell_bw = tf.contrib.rnn.BasicLSTMCell(hidden_size_char)
            
            self._output = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, char_input, 
                    sequence_length=word_lengths, dtype=tf.float32)
            _,((_,output_fw), (_,output_bw)) = self._output
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.reshape(output, shape=[s[0], s[1], 2*hidden_size_char])
        
        self.word_embedding = tf.concat([word_input, output], axis=-1)

        with tf.variable_scope('bi-lstm'):
            cell_fw = tf.contrib.rnn.BasicLSTMCell(lstm_unit_num)
            cell_bw = tf.contrib.rnn.BasicLSTMCell(lstm_unit_num)
            (output_fw, output_bw),_ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.word_embedding,
                    sequence_length=self.sequence_length, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
           
        with tf.variable_scope('proj'):
            W = tf.get_variable('proj_w', [2*lstm_unit_num, tags_num])
            b = tf.get_variable('proj_b', [tags_num])
            x_flat = tf.reshape(output, [-1, 2*lstm_unit_num])
            proj = tf.matmul(x_flat, W) + b
        self.proj = proj 
        self.outputs = tf.reshape(proj, [self.batch_size, -1, tags_num])
        self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.outputs,
                self.target, self.sequence_length)
        
        self.loss = tf.reduce_mean(-self.log_likelihood)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        self.global_step = tf.Variable(0, trainable=False)
        self.increment_step_op = tf.assign(self.global_step, self.global_step+1)
        
        tf.summary.scalar('loss', self.loss)
        self.merged_summary = tf.summary.merge_all()
        

