"""
 python ner.py --action label --input_file /home/zuoguangsheng/bishe/pipeline/txt_utils/data/threatpost_content_txt.processed.small --output_file /home/zuoguangsheng/bishe/pipeline/txt_utils/data/threatpost_content_txt.processed.small.label
"""
import os
import tensorflow as tf
from pipeline.NER.data.gen_train_data import gen_train_data
from pipeline.NER.data.gen_word_embedding import gen_word_embedding

from pipeline.NER import config

tf.app.flags.DEFINE_string("action", 'predict', "train | predict | evaluate | label")
# tf.app.flags.DEFINE_string('process', 'label', 'build|train|evaluate|label')
tf.app.flags.DEFINE_string('checkpoint', '~/bishe/pipeline/NER/new_model/char_lstmcrf_security300_v2/', 'model checkpoint file path') # nd means new_dataset, lg means large word vocab embedding
tf.app.flags.DEFINE_string('model', 'char_lstmcrf', 'lstmcrf|char_lstmcrf')
tf.app.flags.DEFINE_string('input_file', '', 'label input file.')
tf.app.flags.DEFINE_string('output_file','', 'label output file.')
flags = config.FLAGS

from pipeline.NER.build_word_index import build_word_index
from pipeline.NER.lstmcrf import lstmcrf_net
from pipeline.NER.char_lstmcrf import char_lstmcrf_net
from pipeline.NER.utils import prepare_data

cur_path = os.path.dirname(os.path.abspath(__file__))
data_path = cur_path+'/data/'
print (cur_path, data_path)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
def build_ner_context(corpus_file=data_path+'/nvd_corpus.txt', ):
    with open(corpus_file, 'r') as f:
        gen_train_data(f, data_path)
    print ("generating word embedding...")
    gen_word_embedding(data_path)
    build_word_index(gensim_type=True)

def train(model='lstmcrf'):
    net = None 

    if model== 'lstmcrf':
        net = lstmcrf_net("ner")
    elif model== 'char_lstmcrf':
        net = char_lstmcrf_net("char_lstmcrf_net")
    with tf.Session(config=config) as sess:
        net.train(sess)

def evaluate(model='lstmcrf'):
    net = None
    #iterator, embedding, tag_table = prepare_data(action='evaluate', input_file=flags.evaluate_file, batch_size=flags.eval_batch_size)
    if model== 'lstmcrf':
        net = lstmcrf_net('ner')
    elif model== 'char_lstmcrf':
        net = char_lstmcrf_net("char_lstmcrf_net")

    with tf.Session(config=config) as sess:
        eval_res_file = os.path.expanduser(flags.evaluate_res_file) 
        net.evaluate(sess, eval_res_file)

def label(input_file, model='lstmcrf', output_file=flags.output_file):
    '''
    # it haven't support sentence input, only file for now. 
     @ given a setence, label its entity words and entity type.
     @ return a list of pairs, which a pair is such thing like (entity words, entity type, position)
    ''' 
    net = None
    #iterator, embedding, tag_table = prepare_data(action='predict', input_file=input_file, batch_size=flags.pred_batch_size)
    if model == 'lstmcrf':
        net = lstmcrf_net('ner')
    elif model== 'char_lstmcrf':
        net = char_lstmcrf_net("char_lstmcrf_net")

    with tf.Session() as sess:
        output_file = os.path.expanduser(flags.output_file)
        net.predict(sess, input_file, res_file=output_file)

def label_sentence(sentence, model='char_lstmcrf'):
    """label sentence using one by one.
    """
    net = None
    # iterator, embedding, tag_table = prepare_data(action='predict', input_file=input_file, batch_size=flags.pred_batch_size)
    if model == 'lstmcrf':
        net = lstmcrf_net('ner')
    elif model == 'char_lstmcrf':
        net = char_lstmcrf_net("char_lstmcrf_net")

    sent_file = "~/bishe/pipeline/sent_file"

    with open(os.path.expanduser(sent_file), 'w') as f:
        sentence = _preprocess(sentence)
        f.write(sentence + '\n')

    with tf.Session() as sess:
        tag_res = net.predict_one(sess, sent_file)
        #print tag_res
    return sentence, tag_res

def _preprocess(sentence):
    words = sentence.strip().split()
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
    _desc = ""
    for ch in desc:
        if ord(ch) >=128:
            continue
        _desc += ch
    print _desc, "#"*30
    return _desc




if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)

    if flags.action == 'build':
        build_ner_context()
    if flags.action == 'train':
        train(model=flags.model)
    if flags.action == 'evaluate':
        evaluate(model=flags.model)
    if flags.action == 'label':
        label(flags.input_file, model=flags.model)
