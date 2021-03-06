#encoding:utf-8
from __future__ import print_function
import re
from string import lowercase, lower
from six import string_types, text_type, itervalues
from collections import defaultdict
from operator import add
from os import path, walk, makedirs

from gensim.utils import RULE_DISCARD
from gensim.utils import RULE_KEEP
from gensim.utils import RULE_DEFAULT
from gensim.models import word2vec
from config import *

import logging
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)
logger = logging.getLogger(__name__)


_delchars = [chr(c) for c in range(256)]
_delchars = [x for x in _delchars if not x.isalnum()]
_delchars.remove('\t')
_delchars.remove(' ')
_delchars.remove('-')
_delchars.remove('_')  # for instance phrases are joined in word2vec used this char
_delchars = ''.join(_delchars)
_delchars_table = dict((ord(char), None) for char in _delchars)

def standardize_string(s, clean_words=True, lower=True, language="english"):
    """
    Ensures common convention across code. Converts to utf-8 and removes non-alphanumeric characters

    Parameters
    ----------
    language: only "english" is now supported. If "english" will remove non-alphanumeric characters

    lower: if True will lower string.

    clean_words: if True will remove non alphanumeric characters (for instance '$', '#' or 'ł')

    Returns
    -------
    string: processed string
    """

    assert isinstance(s, string_types)

    if not isinstance(s, text_type):
        s = text_type(s, "utf-8")
    if language == "english":
        s = re.sub('<[^>]*>', '', s) #eleminate all xml tags
        s = (s.lower() if lower else s)
        s = (s.translate(_delchars_table) if clean_words else s)
        return s
    else:
        raise NotImplementedError("Not implemented standarization for other languages")


pass_rule = re.compile("[a-zA-Z]+[a-zA-Z0-9'_-]*")

stoplist = STOPLIST

def tr_rule(word, count, min_count):
    if pass_rule.match(word) is None:
        return RULE_DISCARD
    return RULE_DEFAULT


def dict_from_file(path_to_dict):
    vocablist = []
    with open(path_to_dict) as f:
        for word in f.readlines():
            vocablist.append(word)
    if len(vocablist) == 0:
        logger.warn("no word found, size of vocab set is 0")
        return set()
    vocablist = [word.lower() for word in vocablist]
    if vocablist[0][-1] not in lowercase:
        vocablist = [word[:-1] for word in vocablist]
    logger.info("set of vocab finished, size = %d" % len(vocablist))
    return set(vocablist)


def filter_walk(data_folder, ext_name = '.html'):
    for base_path, dirs, files in walk(data_folder):
        for filename in files:
            fullpath = path.join(base_path, filename)
            filename, file_extension = path.splitext(fullpath)
            if file_extension == ext_name:
                yield fullpath
    

def file2sentences(filename, stoplist, dictionary):
    sentences = []
    with open(filename) as f:
        sentences = f.readlines()
    if len(sentences) == 0:
        return [[]]
    sentences = reduce(add, sentences)
    sentences = [[word for word in sentence.lower().split() if word not in stoplist and word in dictionary] for sentence in sentences.split('.')]
    return sentences


def folder2sentences(corpora_folder, num_docs, dictionary):
    enum_files = enumerate(filter_walk(corpora_folder))
    for file_no, filename in enum_files:
        if (file_no+1)%200 == 0:
            logger.info('retrieving doc number %d' % (file_no+1))
        yield file2sentences(filename, stoplist, dictionary)
        if (file_no == num_docs-1):
            return


def scan_vocab_custom(model, sentences, dictionary,
                      progress_per=10000, trim_rule=None, update=False):
    sentence_no = -1
    total_words = 0
    min_reduce = 1
    if model.raw_vocab:
        vocab = model.raw_vocab
    else:
        vocab = defaultdict(int)

    checked_string_types = 0
    for sentence_no, sentence in enumerate(sentences):
        if not checked_string_types:
            if isinstance(sentence, string_types):
                logger.warn("Each 'sentences' item should be a list of words (usually unicode strings)."
                    "First item here is instead plain %s.", type(sentence))
                checked_string_types += 1
        
        if dictionary is not None:
            for word in (lower(word) for word in sentence if word in dictionary):
                vocab[word] += 1
        else:
            for word in (word for word in sentence):
                vocab[word] += 1

        #if model.max_vocab_size and len(vocab) > model.max_vocab_size:
        #    total_words += prune_vocab(vocab, min_reduce, trim_rule=trim_rule)
        #    min_reduce += 1

        total_words += sum(itervalues(vocab))
        
        if hasattr(model, 'corpus_count'):
            model.corpus_count += sentence_no + 1
        else:
            model.corpus_count = sentence_no + 1
            
        if not model.raw_vocab:
            model.raw_vocab = vocab


def err (model, x, y, topn = 20):
    top20 = model.most_similar(positive = [x], topn = topn)
    rank = 1
    if not y in dict(top20):
        rank = 2
    else:
        rank = dict(zip(map(lambda a: a[0], top20), [i for i in range(1,21)]))[y]
    return (1.0/rank)


def folderfy(folder_name):
    if not path.exists(folder_name):
        makedirs(folder_name)
    if folder_name[-1]!='/':
        folder_name+='/'
    return folder_name


def retrieve_models(models_folder, basename, file_suffix, num_models=1):
    if num_models==1:
        return word2vec.Word2Vec.load(models_folder+basename + '.w2v' + file_suffix)
    else:
        return [word2vec.Word2Vec.load(models_folder+basename + str(idx) + '.w2v' + file_suffix) for idx in range(1, num_models+1)]
