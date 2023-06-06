# -*- coding: utf-8 -*
import os
import traceback
from tensorflow.contrib.rnn import DropoutWrapper
from pipeline.NER.utils import *
from pipeline.NER import config

BATCH_SIZE = config.FLAGS.batch_size
unit_num = embeddings_size    
time_step = max_sequence     
DROPOUT_RATE = config.FLAGS.dropout
EPOCH = config.FLAGS.epoch
evaluate_file = config.FLAGS.evaluate_file
model_path = config.FLAGS.model_path

cur_path = os.path.abspath(os.path.dirname(__file__)) 

flags = tf.app.flags.FLAGS

epoch_checkpoint_file = "~/bishe/pipeline/NER/epochs_model/charlstm_epoch_%d/"

class lstmcrf_net(object):
    def __init__(self, scope_name):
        '''
        :param scope_name:
        :param iterator: 调用tensorflow DataSet API把数据feed进来。
        :param embedding: 提前训练好的word embedding
        :param batch_size:
        '''
        self.embedding, self.tag_table, self.src_vocab_table, self.tgt_vocab_table, self.vocab_size = prepare_data_x()
        print ("prepare data done.")

        with tf.variable_scope(scope_name) as scope:
            self._build_net()


    def _build_net(self):

        self.source = tf.placeholder(tf.int32, shape=([None, None]))
        self.target = tf.placeholder(tf.int32, shape=([None, None]))
        self.sequence_length = tf.placeholder(tf.int32, shape=([None, ]))
        self.char_source = tf.placeholder(tf.int32, shape=([None, None, None]))
        self.word_lengths = tf.placeholder(tf.int32, shape=([None, None]))

        self.global_step = tf.Variable(0, trainable=False)
        TAGS_NUM = get_class_size()
        self.batch_size = tf.shape(self.source)[0]

        # x: [batch_size, time_step, embedding_size], float32
        self.x = tf.nn.embedding_lookup(self.embedding, self.source)
        # y: [batch_size, time_step]
        self.y = self.target

        cell_forward = tf.contrib.rnn.BasicLSTMCell(unit_num)
        cell_backward = tf.contrib.rnn.BasicLSTMCell(unit_num)
        if DROPOUT_RATE is not None:
            cell_forward = DropoutWrapper(cell_forward, input_keep_prob=1.0, output_keep_prob=DROPOUT_RATE)
            cell_backward = DropoutWrapper(cell_backward, input_keep_prob=1.0,  output_keep_prob=DROPOUT_RATE)

        rnn_outputs, bi_state = \
            tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.x, sequence_length=self.sequence_length, dtype=tf.float32)

        forward_out, backward_out = rnn_outputs
        self.rnn_outputs = tf.concat([forward_out, backward_out], axis=2)

        # projection:
        W = tf.get_variable("projection_w", [2 * unit_num, TAGS_NUM])
        b = tf.get_variable("projection_b", [TAGS_NUM])
        x_reshape = tf.reshape(self.rnn_outputs, [ -1, 2 * unit_num])
        projection = tf.matmul(x_reshape, W) + b

        # -1 to time step
        self.outputs = tf.reshape(projection, [self.batch_size, -1, TAGS_NUM])
        self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            self.outputs, self.y, self.sequence_length)

        # Add a training op to tune the parameters.
        self.loss = tf.reduce_mean(-self.log_likelihood)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        self.increment_step_op = tf.assign(self.global_step, self.global_step+1) 
        
        tf.summary.scalar('loss',self.loss)
        self.merged_summary = tf.summary.merge_all()

    def train(self, sess, continue_train=False, epochs_log=False):
        iterator = get_iterator(self.src_vocab_table, self.tgt_vocab_table, self.vocab_size, flags.batch_size)
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        tf.tables_initializer().run()
        saver = tf.train.Saver()
        checkpoint_file =os.path.expanduser(flags.checkpoint)
        if not tf.gfile.Exists(checkpoint_file):
            tf.gfile.MkDir(checkpoint_file)
        log_writer = tf.summary.FileWriter(checkpoint_file, sess.graph)
        print ("checkpoint file path:%s"%checkpoint_file)
        epoches = 0
        old_epoches = 0 
        while True:
            if epoches > old_epoches and epochs_log:
                
                e_checkpoint_file =os.path.expanduser(epoch_checkpoint_file%old_epoches)
                print e_checkpoint_file 
                if not tf.gfile.Exists(e_checkpoint_file):
                    tf.gfile.MkDir(e_checkpoint_file)
                e_checkpoint_file = e_checkpoint_file+'/model'
                e_log_writer = tf.summary.FileWriter(e_checkpoint_file, sess.graph)
                saver.save(sess, e_checkpoint_file)
                e_log_writer.add_summary(summary, global_step=current_steps)        

                old_epoches = epoches 

            if epoches>=EPOCH: break
            try:
                #word_embedding = sess.run([self.word_embedding]) 
                #print np.shape(word_embedding)
                src, tgt, src_len, char_src, word_lens = \
                    sess.run([iterator.source,
                                   iterator.target,
                                   iterator.source_sequence_length,
                                   iterator.char_source,
                                   iterator.word_lengths])
                feed_dict = {self.source: src,
                             self.target: tgt,
                             self.sequence_length: src_len,
                             self.char_source: char_src,
                             self.word_lengths: word_lens}

                tf_unary_scores, tf_transition_params, _, _,losses,current_steps,summary= sess.run(
                    [self.outputs, self.transition_params, self.train_op, self.increment_step_op,
                     self.loss, self.global_step, self.merged_summary],
                    feed_dict = feed_dict)
               
                print ("[epoches]:%d \t [steps]: %d \t [loss]: %f"% (epoches,  current_steps,  losses))
                saver.save(sess, checkpoint_file)
                log_writer.add_summary(summary, global_step=current_steps)        

            except tf.errors.OutOfRangeError:
                print ("one epoch end.")

                sess.run(iterator.initializer)
                epoches += 1
            except tf.errors.InvalidArgumentError:
                # iterator.next() cannot get enough data to a batch, initialize it.
                print traceback.format_exc()
                print ("one epoch end.")

                epoches += 1
                sess.run(iterator.initializer)
                break
        print ('training finished!')


    def predict(self, sess, input_file, res_file=None):
        iterator = get_predict_iterator(input_file, self.src_vocab_table, self.vocab_size, flags.pred_batch_size)

        sess.run(iterator.initializer)
        tf.tables_initializer().run()
        saver = tf.train.Saver()
        checkpoint_file = os.path.expanduser(flags.checkpoint)
        file_iter = file_content_iterator(input_file)
        saver.restore(sess, checkpoint_file)
        tags_res = []
        while True:
            try:
                #shapes = sess.run([tf.shape(self.source), tf.shape(self.x), tf.shape(self.rnn_outputs),
                #    tf.shape(self.outputs), tf.shape(self.transition_params)])
                #print shapes
                src, tgt, src_len, char_src, word_lens = \
                    sess.run([iterator.source,
                              iterator.target,
                              iterator.source_sequence_length,
                              iterator.char_source,
                              iterator.word_lengths])
                print src.shape
                feed_dict = {self.source: src,
                             self.target: tgt,
                             self.sequence_length: src_len,
                             self.char_source: char_src,
                             self.word_lengths: word_lens}

                tf_unary_scores, tf_transition_params, batch_size, sequence_length = sess.run(
                    [self.outputs, self.transition_params, self.batch_size, self.sequence_length],
                    feed_dict = feed_dict)
                decode_sequences,_ = sess.run(tf.contrib.crf.crf_decode(tf_unary_scores, tf_transition_params, sequence_length))
                tags = sess.run(self.tag_table.lookup(tf.dtypes.cast(decode_sequences, tf.int64))) 
                for i in range(batch_size):
                    tags_res.append(tags[i,:])
            except tf.errors.OutOfRangeError:
                sess.run(iterator.initializer)
                break
            except tf.errors.InvalidArgumentError as e:
                # iterator.next() cannot get enough data to a batch, initialize it.
                # sess.run(iterator.initializer)
                # epoches += 1
                print (traceback.format_exc(e))
                

        write_result_to_file(file_iter, tags_res, res_file)

    def evaluate(self, sess, eval_res_file):
        evaluate_file = os.path.expanduser(flags.evaluate_file)
        evaluate_tgt_file = os.path.expanduser(flags.evaluate_tgt_file) 
        self.predict(sess, evaluate_file, eval_res_file)
        metrics = compute_evaluate_metrics(evaluate_file, eval_res_file, evaluate_tgt_file)
        print (metrics)


    def predict_one(self, sess, sentence):
        iterator = get_predict_iterator_for_sentence(sentence, self.src_vocab_table, self.vocab_size, 1)
        sess.run(iterator.initializer)
        tf.tables_initializer().run()

        saver = tf.train.Saver()
        checkpoint_file = os.path.expanduser(flags.checkpoint)
        saver.restore(sess, checkpoint_file)

        tags_res = []
        while True:
            try:
                src, tgt, src_len, char_src, word_lens = \
                    sess.run([iterator.source,
                              iterator.target,
                              iterator.source_sequence_length,
                              iterator.char_source,
                              iterator.word_lengths])
                feed_dict = {self.source: src,
                             self.target: tgt,
                             self.sequence_length: src_len,
                             self.char_source: char_src,
                             self.word_lengths: word_lens}

                tf_unary_scores, tf_transition_params, batch_size, sequence_length = sess.run(
                    [self.outputs, self.transition_params, self.batch_size, self.sequence_length],
                    feed_dict = feed_dict)
                decode_sequences,_ = sess.run(tf.contrib.crf.crf_decode(tf_unary_scores, tf_transition_params, sequence_length))
                tags = sess.run(self.tag_table.lookup(tf.dtypes.cast(decode_sequences, tf.int64)))
                for i in range(batch_size):
                    tags_res.append(tags[i,:])
            except tf.errors.OutOfRangeError:
                sess.run(iterator.initializer)
                break
            except tf.errors.InvalidArgumentError as e:
                # iterator.next() cannot get enough data to a batch, initialize it.
                # sess.run(iterator.initializer)
                # epoches += 1
                print (traceback.format_exc(e))
        return tags_res[0]


if __name__ == '__main__':

    action = config.FLAGS.action
    # 获取词的总数。
    vocab_size = get_src_vocab_size()
    src_unknown_id = tgt_unknown_id = vocab_size
    src_padding = vocab_size + 1

    src_vocab_table, tgt_vocab_table = create_vocab_tables(vocab_file, label_file, src_unknown_id,
                                                           tgt_unknown_id)
    embedding = load_word2vec_embedding(vocab_size)
    
    print ("preparing dataset iterator...")
    if action == 'train':
        iterator = get_iterator(src_vocab_table, tgt_vocab_table, vocab_size, BATCH_SIZE)
    elif action == 'predict':
        BATCH_SIZE = 1
        DROPOUT_RATE = 1.0
        iterator = get_predict_iterator(src_vocab_table, vocab_size, BATCH_SIZE)
    else:
        print ('Only support train and predict actions.')
        exit(0)

    tag_table = tag_to_id_table()
    net = lstmcrf_net("ner", iterator, embedding, BATCH_SIZE)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        tf.tables_initializer().run()
        if action == 'train':
            train(net, iterator, sess)
        elif action == 'predict':
            predict(net, tag_table, sess)
