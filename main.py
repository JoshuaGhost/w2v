from __future__ import print_function, division

from time import time
from copy import deepcopy
from os import path

import pickle
import argparse

from gensim.models import word2vec

from utils import *
from Evaluator import Evaluator
from Combiner import Combiner

def collect_sentences(corpora_folder, num_all_docs, ce_folder, valid_vocab):
	timestamp = -time()
	
	sentences = []
	for s in folder2sentences(corpora_folder, num_all_docs, valid_vocab):\
		sentences += s
	timestamp += time()
	with open(ce_folder+'/time.txt', 'a+') as f:
		f.write('%d docs in total, sentences collection time: %0.6f\n\n'% (num_all_docs, timestamp))
	return sentences


def w2v_timing(sentences, ebd_dim, min_count, model_suffix, test_mode, ce_folder):
	timestamp = -time()
	model = word2vec.Word2Vec(sentences = sentences, size = ebd_dim, window = 10, min_count = min_count,
					max_vocab_size = MAX_VOCAB_SIZE, sample = 1e-4, workers = NUM_WORKERS, 
					sg = 1, negative = 5, iter = NUM_ITER, null_word = 1,
					trim_rule = tr_rule, batch_words = BATCH_WORDS)
	model_filename = 'dim_%d_mc_%d_%s.w2v'% (ebd_dim, min_count, model_suffix)
	model_filename += '.test' if test_mode else ''

	timestamp += time()
	with open(ce_folder+'time.txt', 'a+') as f:
		f.write("ebd_dim %d min_count %d training time:\n %0.6f\n"% (ebd_dim, min_count, timestamp))
	return model, model_filename
			

def build_model_vocab_only(corpora_folder, num_all_docs, ce_folder, dictionary,
			   ebd_dim, min_count, models_folder, corpora_file = None):
    
    model = word2vec.Word2Vec(size=ebd_dim, window=10, max_vocab_size=MAX_VOCAB_SIZE,\
                              sample=1e-4, workers=NUM_WORKERS, sg=1, negative=5, trim_rule = tr_rule,\
                              iter=NUM_ITER, null_word=1, batch_words = BATCH_WORDS)
    
    model.raw_vocab = {}
    model.corpus_count = 0

    sentences = []
    if corpora_file is not None:
        for idx, line in enumerate(corpora_file):
            for s in standardize_string(line).split('.'):
                if len(s)>0:
                    sentences.append(s.split())
            if test_mode and len(sentences)>100:
                scan_vocab_custom(model, sentences, None, logger)
                sentences = []
                break
            if len(sentences) >= 1000000:#collect 1000000 sentences for vocabulary building
                scan_vocab_custom(model, sentences, None, logger)
                sentences = []
        if (idx+1)%1000000 == 0:
            logger.info('collected %d sentences with %d raw_vocab'%(sent_idx+1, len(model.raw_vocab)))
    else:
        sent_gen = collect_sentences(corpora_folder, num_all_docs, ce_folder, dictionary)
        for sent_idx, sentences in enumerate(sent_gen):
            scan_vocab_custom(model, sentences, dictionary, logger)
            if (sent_idx+1)%2000 == 0:
                logger.info('collected '+str(sent_idx+1)+' sentences with '+str(len(model.raw_vocab))+' raw_vocab')

    model.scale_vocab(min_count=min_count, trim_rule=tr_rule)
    model.finalize_vocab()
    model.save(models_folder + 'model_vocab_only_ebd_%d_mc_%d.w2v'% (ebd_dim, min_count))
    return model


exp_discription = {0: "don't generate models, evaluate only",
				   1: "iterating min_count from MIN_COUNT_MIN to MIN_COUNT_MAX for MIN_COUNT_STEP, ebd_dim from DIM_MIN to DIM_MAX for DIM_STEP",
				   2: "specify ebd_dim and min_count",
				   3: "generate num_sub_models sub modules using shared vocabulary, combined using sorting",
				   4: "generate num_sub_models sub modules using shared vocabulary, combined using lsr"}


parser = argparse.ArgumentParser(description = 'Parser of main entry')

parser.add_argument('-c', type=str, help='dirctory of corpora')
parser.add_argument('-m', type=str, help='directory of models')
parser.add_argument('-t', type=int, help='''determine types of training:
											0: %s
											1: %s
											2: %s
											3: %s
											4: %s''' % tuple(exp_discription.values()))
parser.add_argument('-b', type=str, help='directory of benchmark')
parser.add_argument('-d', type=str, help='path to dictionary file')
parser.add_argument('-f', type=int, help='''determine format of benchmark:
											0: wordnet and wikipedia''')
parser.add_argument('-r', type=str, help='directory of chronological and evaluation result')
parser.add_argument('--ebd_dim', type=int, help='number of embedding dimension')
parser.add_argument('--min_count', type=int, help='threshold of word count frequency')
parser.add_argument('--num_sub_models', type=int, help='number of sub models')
parser.add_argument('--test_mode', action='store_const', const=True, help='if set, then run under test mode(smaller amount of total docs)')

args = vars(parser.parse_args())


exp_type		= args['t'] if args['t'] is not None else 0
benchmark_form	= args['f'] if args['f'] is not None else 0

if not exp_type==0:
	corpora_folder		= folderfy(args['c'] if args['c'] is not None else CORPORA_FOLDER)
models_folder 		= folderfy(args['m'] if args['m'] is not None else MODELS_FOLDER)
benchmark_folder 	= folderfy(args['b'] if args['b'] is not None else BENCHMARK_FOLDER)
ce_folder			= folderfy(args['r'] if args['r'] is not None else CE_FOLDER)

dict_path			= args['d'] if args['d'] is not None else DICT_PATH

test_mode 		= args['test_mode'] if args['test_mode'] is not None else TEST_MODE

ebd_dim 		= args['ebd_dim'] if args['ebd_dim'] is not None and args['ebd_dim'] is not None else EBD_DIM_DEFAULT
min_count 		= args['min_count'] if args['min_count'] is not None and args['min_count'] is not None else MIN_COUNT_DEFAULT

num_all_docs 	= NUM_ALL_DOCS_TEST if test_mode else NUM_ALL_DOCS
num_sub_models 	= args['num_sub_models'] if args['num_sub_models'] is not None and args['num_sub_models'] is not None else NUM_SUB_MODEL_DEFAULT


evaluator = Evaluator.eval_factory(benchmark_form, ce_folder, benchmark_folder,\
			exp_type, benchmark_form, num_all_docs,\
			num_sub_models, test_mode)

if test_mode:
	file_suffix='.test'
else:
	file_suffix=''

with open(ce_folder+'/time.txt', 'a+') as f:
	f.write('='*24+'\nexperiment type: %d\n'%exp_type)
	f.write('configuration:\n')
	f.write('\tis test mode: %d\n'% test_mode)
	f.write('\tnum of all docs: %d\n'% num_all_docs)


if exp_type==0:
	for model_filename in filter_walk(models_folder, ('.test' if test_mode else '.w2v')):
		print( model_filename)
		model = word2vec.Word2Vec.load(model_filename)
		ebd_dim = int(model_filename.split('_')[-4])
		min_count = int(model_filename.split('_')[-2])
		evaluator.eval_ext(model, ebd_dim, min_count)


elif exp_type==1:
	valid_vocab = dict_from_file(dict_path)
	sentences = collect_sentences(corpora_folder, num_all_docs, ce_folder, valid_vocab)

	for ebd_dim in range(DIM_MIN, DIM_MAX, DIM_STEP):
		for min_count in range(MIN_COUNT_MIN, MIN_COUNT_MAX, MIN_COUNT_STEP):
			model, model_filename = w2v_timing(sentences, ebd_dim, min_count, 'iter', test_mode, ce_folder)
			evaluator.eval_ext(model, ebd_dim, min_count)
			model.save(models_folder+model_filename+file_suffix)


elif exp_type==2:
	dictionary = dict_from_file(dict_path)
	sentences = collect_sentences(corpora_folder, num_all_docs, ce_folder, dictionary)
        print (sentences)
	model, model_filename = w2v_timing(sentences, ebd_dim, min_count, 'spec', test_mode, ce_folder)
	
	evaluator.eval_ext(model, ebd_dim, min_count)
	model.save(models_folder+model_filename+file_suffix)


elif exp_type==3:
	dictionary = dict_from_file(dict_path)
	timestamp = -time()
	model_vocab_only = build_model_vocab_only(corpora_folder, num_all_docs, ce_folder, dictionary, ebd_dim, min_count, models_folder)
	timestamp += time()
	with open(ce_folder+'time.txt', 'a+') as f:
		f.write("ebd_dim %d min_count %d base model for vocab sharing:\n %0.6f\n"% (ebd_dim, min_count, timestamp))	
			
	sent_gen = collect_sentences(corpora_folder, num_all_docs, ce_folder, dictionary)
	sentences = []
	for doc_idx, sentences_cur in enumerate(sent_gen):
		sentences += sentences_cur
		num_docs_per_model = num_all_docs//num_sub_models
		if ((doc_idx+1)%num_docs_per_model==0 or doc_idx+1==num_all_docs):
			model_num = doc_idx//num_docs_per_model
			model_tmp = deepcopy(model_vocab_only)
			print (len(sentences))
			timestamp = -time()
			model_tmp.train(sentences, total_examples=len(sentences))
			model_tmp.save(models_folder + 'sub_model_ebd_%d_mc_%d_%d.w2v%s'% (ebd_dim, min_count, model_num, file_suffix))
			timestamp += time()
			with open(ce_folder+'time.txt', 'a+') as f:
				f.write("ebd_dim %d min_count %d training time:\n %0.6f\n" % (ebd_dim, min_count, timestamp))	
			model_tmp = None
			sentences = []

	num_cbn_sample_words = NUM_CBN_SAMPLE_WORDS
	combiner = Combiner.cbn_factory(0, num_cbn_sample_words, True, test_mode)
	models = retrieve_models(models_folder, 'sub_model_ebd_%d_mc_%d_'% (ebd_dim, min_count), file_suffix, num_sub_models)
	model_combined = combiner.combine(models)
	evaluator.eval_ext(model_combined, ebd_dim, min_count)
	model_filename=("combined_model_ebd_%d_mc_%d.w2v"% (ebd_dim, min_count))+file_suffix
	model_combined.save(models_folder+model_filename+file_suffix)


elif exp_type==4:
	print("not implesmented yet")

elif exp_type==5:#based on the paper Iproving distributional similarity with lessons learned from word embeddings from Omen Levy
    sentences = []
    ebd_dim = 500
    min_count = 100
    import gzip as gz
    vocab_folder = 'vocab/'
    with open(vocab_folder+'vocab.txt', 'r') as f:
        timestamp = -time()
	model = build_model_vocab_only(corpora_folder, 1, ce_folder,\
                                       None, ebd_dim, 1, models_folder,\
                                       corpora_file = f)
	timestamp += time()
    
    from pdb import set_trace as bp
    bp()
    with open(ce_folder+'time.txt', 'a+') as f:
        f.write('building vocab with min_count %d within time:\n %0.6f\n'% (min_count, timestamp))

    model_filename = 'dim_%d_mc_%d_%s.w2v'% (ebd_dim, min_count, 'wiki')
    model_filename += '.test' if test_mode else ''
   
    sentences_count = 0L
    train_per = 1000000000
    with gz.open(corpora_folder+'wiki_clean_2014.csv.gz', 'rb') as f:
        for line in f:
            for s in standardize_string(line).split('.'):
                if len(s)>0:
                    sentences.append(s.split())
                    if (len(sentences)+1) % (train_per//100) ==0:
                        logger.info('{}% centences of a training batch has been collected'.format((len(sentences)+1)//(train_per//100)))
            if test_mode and len(sentences)>100:
                break
            if len(sentences)>0 and (len(sentences)+1) % train_per == 0:#train every train_per sentences
                logger.info('training...')
                timestamp -= time()
                model.train(sentences, total_examples = len(sentences))
                timestamp += time()
                sentences = []
                logger.info('training complete')
                sentences_count += 1L
    if len(sentences) > 0:
        timestamp -= time()
        model.train(sentences, total_examples = len(sentences))
        timestamp += time()
        sentences_count = sentences_count * train_per + len(sentences)
        sentences = []

    with open(ce_folder+'time.txt', 'a+') as f:
	f.write("ebd_dim %d min_count %d training time:\n %0.6f\n"% (ebd_dim, min_count, timestamp))
        f.write("intotal %d sentences are used for training\n\n"% sentences_count)
    model.save(models_folder+model_filename)

