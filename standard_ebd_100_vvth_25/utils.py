from string import lowercase, lower
from six import string_types, itervalues
from collections import defaultdict
import os
import re
from gensim.utils import RULE_DISCARD
from gensim.utils import RULE_KEEP
from gensim.utils import RULE_DEFAULT

prog = re.compile("[a-zA-Z]+[a-zA-Z0-9'_-]*")


def vocab_set_from_dict(path_to_dict):
	vocab_list = []
	with open(path_to_dict) as f:
		for word in f.readlines():
			vocab_list.append(word.lower())
	if len(vocab_list) == 0:
		return set()

	if vocab_list[0][-1] not in lowercase:
		vocab_list = [word[:-1] for word in vocab_list]

	return set(vocab_list)


def scan_vocab_custom(model, sentences, dictionary, logger,
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
		
		for word in (lower(word) for word in sentence if word in dictionary):
			vocab[word] += 1

		if model.max_vocab_size and len(vocab) > model.max_vocab_size:
			total_words += prune_vocab(vocab, min_reduce, trim_rule=trim_rule)
			min_reduce += 1

		total_words += sum(itervalues(vocab))
		
		if hasattr(model, 'corpus_count'):
			model.corpus_count += sentence_no + 1
		else:
			model.corpus_count = sentence_no + 1
			
		if not model.raw_vocab:
			model.raw_vocab = vocab


def walk_all_files(data_folder, ext_name = '.html'):
    for path, dirs, files in os.walk(data_folder):
        for filename in files:
            fullpath = os.path.join(path, filename)
            filename, file_extension = os.path.splitext(fullpath)
            if file_extension == ext_name:
                yield fullpath


def gen_sentence(data_folder, num_docs, stoplist, dictionary, test_mode = False, num_docs_test = 100):
	enum_files = enumerate(walk_all_files(data_folder))
	for file_num, filename in enum_files:
		with open(filename) as f:
			sentences = f.readlines()
			if len(sentences) == 0:
				yield []
				continue
			sentences = reduce(lambda x, y: x+y, sentences)
			for sentence in sentences.split('.'):
				yield [word for word in sentences.lower().split() if word not in stoplist and word in dictionary]
		if (file_num == num_docs-1) or (test_mode and file_num == num_docs_test-1):
			return


def tr_rule(word, count, min_count):
	if prog.match(word) is None:
		return RULE_DISCARD
	return RULE_DEFAULT
