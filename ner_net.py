
import os
import tensorflow as tf
from pipeline.NER.data.gen_train_data import gen_train_data
from pipeline.NER.data.gen_word_embedding import gen_word_embedding

from pipeline.NER import config

tf.app.flags.DEFINE_string("action", 'predict', "train | predict | evaluate | label")
# tf.app.flags.DEFINE_string('process', 'label', 'build|train|evaluate|label')
tf.app.flags.DEFINE_string('checkpoint', '~/bishe/pipeline/NER/new_model/char_lstmcrf_security300/', 'model checkpoint file path') # nd means new_dataset, lg means large word vocab embedding
tf.app.flags.DEFINE_string('model', 'char_lstmcrf', 'lstmcrf|char_lstmcrf')
tf.app.flags.DEFINE_string('input_file', '', 'label input file.')
tf.app.flags.DEFINE_string('output_file','', 'label output file.')
flags = config.FLAGS

from pipeline.NER.lstmcrf import lstmcrf_net
from pipeline.NER.char_lstmcrf import char_lstmcrf_net

cur_path = os.path.dirname(os.path.abspath(__file__))
data_path = cur_path+'/data/'
print (cur_path, data_path)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

class ner_net(object):
    """
    provides api class.
    """
    def __init__(self, model_name='char_lstmcrf'):
        self.model = None
        if model_name =="char_lstmcrf":
            self.model = char_lstmcrf_net(model_name)
        elif model_name == "lstmcrf":
            self.model = lstmcrf_net(model_name)
    def label(self, sentence):
        with tf.Session(graph=tf.get_default_graph()) as sess:
            sentence = self._preprocess(sentence)
            sent_file = self._write_to_file(sentence)
            tag_res = self.model.predict_one(sess,sent_file)
            print (tag_res)
        return tag_res

    def _write_to_file(self, sentence):
        sent_file = "~/bishe/pipeline/sent_file"

        with open(os.path.expanduser(sent_file),'w') as f:
            for i in range(5):
                f.write(sentence+'\n')
        return sent_file

    def _preprocess(self, sentence):
        words = sentence.split()
        desc = ""
        for word in words:
            word = word.lower()
            if word[0] == '(' and word[-1] == ')':
                desc += word + " "
            elif word[0] in [',', '"', '.', ';', '(']:
                desc += str(word[0]) + " " + str(word[1:]) + " "
            elif word[-2:] == '."':
                desc += str(word[:-2]) + " " + str[word[-1]] + " "
            elif word[-1] in [',', '"', '.', ';', ')']:
                desc += str(word[:-1]) + " " + str(word[-1]) + " "
            else:
                desc += word + " "
        return desc