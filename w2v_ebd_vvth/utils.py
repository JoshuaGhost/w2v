import os
import re
from gensim.utils import RULE_DISCARD
from gensim.utils import RULE_KEEP
from gensim.utils import RULE_DEFAULT

prog = re.compile("[a-zA-Z]+[a-zA-Z0-9'_-]*")

def walk_all_files(data_folder, ext_name = '.html'):
    for path, dirs, files in os.walk(data_folder):
        for filename in files:
            fullpath = os.path.join(path, filename)
            filename, file_extension = os.path.splitext(fullpath)
            if file_extension == ext_name:
                yield fullpath

def read_sentences_from(f, prune_list):
	sentences = f.readlines()
	if len(sentences) == 0:
		return [[]]
	sentences = reduce(lambda x, y: x+y, sentences)
	sentences = [[word for word in sentence.lower().split() if word not in prune_list] for sentence in sentences.split('.')]
	return sentences

def tr_rule(word, count, min_count):
	if prog.match(word) is None:
		return RULE_DISCARD
	return RULE_DEFAULT
