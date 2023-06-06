#encoding=utf-8
import os
from gensim.models import KeyedVectors
from pipeline.NER import config

vocab_file = os.path.expanduser(config.FLAGS.vocab_file)
label_file = os.path.expanduser(config.FLAGS.label_file)
train_tgt_file = os.path.expanduser(config.FLAGS.train_tgt_file)
word_embedding_file = config.FLAGS.word_embedding_file

def build_word_index(gensim_type=True):
    '''
        生成单词列表，并存入文件之中。
    :return:
    '''
    if not os.path.exists(word_embedding_file):
        print ('word embedding file does not exist, please check your file path ')
        return

    print ('building word index...')
    if not os.path.exists(vocab_file): #always rewrite vocab file
        if gensim_type:
            wv = KeyedVectors.load(word_embedding_file, mmap='r')
            words=wv.vocab.keys()
            f = open(vocab_file,'w')
            for word in words:
                # if type(word) is unicode:
                #     word = word.encode('utf-8')
                f.write(word+'\n')
            f.close()
        else:
            print vocab_file
            with open(vocab_file, 'w') as source:
                f = open(word_embedding_file, 'r')
                for line in f:
                    values = line.split()
                    word = values[0]  # 取词
                    # if type(word) is unicode:
                    #     word = word.encode('utf8')
                    source.write(word + '\n')
            f.close()
    else:
        print ('source vocabulary file has already existed, continue to next stage.')

    if not os.path.exists(label_file):
        print ("no label file now.")
        with open(train_tgt_file, 'r') as source:
            dict_word = {}
            # with open('source_vocab', 'w') as s_vocab:
            for line in source.readlines():
                line = line.strip()
                if line != '':
                    word_arr = line.split()
                    for w in word_arr:
                        dict_word[w] = dict_word.get(w, 0) + 1

            top_words = sorted(dict_word.items(), key=lambda s: s[1], reverse=True)
            with open(label_file, 'w') as s_vocab:
                for word, frequence in top_words:
                    s_vocab.write(word + '\n')
    else:
        print ('target vocabulary file has already existed, continue to next stage.')

#build_word_index(gensim_type=False)

def diff_with_glove():
    with open("data/train.txt", 'r') as source:
        dict_word = {}
        for line in source.readlines():
            line = line.strip()
            if line != '':
                word_arr = line.split()
                for w in word_arr:
                    dict_word[w] = dict_word.get(w,0) + 1
    with open(vocab_file,'r') as f:
        gloves = {}
        for line in f.readlines():
            line = line.strip()
            if line != '':
               gloves[line] = 1 
    print (len(gloves))
    count = 0 
    for word,req in dict_word.items():
        word = word.lower()
        if word not in gloves:
            count += 1
            print (word, req)
    print (count, len(dict_word))

#diff_with_glove()
if __name__ == '__main__':
   build_word_index(gensim_type=False)
