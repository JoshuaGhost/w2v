PYTHON_INTERPRETER=python
MAIN_ENTRY=main.py
DUMP_PWD=$(shell pwd)
MODELS_FOLDER=$(DUMP_PWD)/result/default_models/
USERNAME=$(shell whoami)
CORPORA_FOLDER=/home/$(USERNAME)/workspace/master_thesis/data/
#CORPORA_FOLDER=/data1/alexandria/
BENCHMARK_FOLDER=/home/$(USERNAME)/workspace/master_thesis/benchmark/EN-WORDREP
CE_FOLDER=$(DUMP_PWD)/result/default_results/
DICT_PATH=/usr/share/dict/words
BENCHMARKS_WORD_EMBEDDING=./benchmarks
WIKI_STANDARD_MODEL=$(MODELS_FOLDER)wiki_standard/dim_500_wiki.w2v
#WIKI_STANDARD_MODEL=$(DUMP_PWD)/../models/new/dim_25_mc_0_iter.w2v
	
TEST ?= 0
ifeq ($(TEST), 1)
	TEST_OPT=--test_mode
endif

help :
	@echo 'Makefile for Word2Vec experiment'
	@echo ''
	@echo 'Usage:'
	@echo '		make iter       train multiple models using various of dimension and min count'
	@echo "		make eval       don't train any models, just evaluate"
	@echo '		make sort_comb  train dedicated models and combine them using sort alignment method'
	@echo '		make lsr_comb   train dedicated models and combine them using LSR'
	@echo '     make spec       train single model with specified configuration, default size=100 min_count=24'
	@echo '     make wiki       use wiki dump as corpora'
	@echo ''


dirs :
	mkdir -p MODELS_FOLDER
	mkdir -p CE_FOLDER
	
eval : dirs
	$(PYTHON_INTERPRETER) $(MAIN_ENTRY) -t 0 -f 0 -m $(MODELS_FOLDER) -b $(BENCHMARK_FOLDER) -r $(CE_FOLDER) $(TEST_OPT)

eval_all : dirs
	cd $(BENCHMARKS_WORD_EMBEDDING); \
	$(PYTHON_INTERPRETER) evaluate_on_all.py -f $(WIKI_STANDARD_MODEL) -p gensim -o $(CE_FOLDER)/eval_wiki_standard.csv

iter : dirs
	$(PYTHON_INTERPRETER) $(MAIN_ENTRY) -t 1 -f 0 -c $(CORPORA_FOLDER) -m $(MODELS_FOLDER) -b $(BENCHMARK_FOLDER) -r $(CE_FOLDER) -d $(DICT_PATH) $(TEST_OPT)

spec : dirs
	$(PYTHON_INTERPRETER) $(MAIN_ENTRY) -t 2 -f 0 -c $(CORPORA_FOLDER) -m $(MODELS_FOLDER) -b $(BENCHMARK_FOLDER) -r $(CE_FOLDER) -d $(DICT_PATH) $(TEST_OPT)

sort_comb : dirs
	$(PYTHON_INTERPRETER) $(MAIN_ENTRY) -t 3 -f 0 -c $(CORPORA_FOLDER) -m $(MODELS_FOLDER) -b $(BENCHMARK_FOLDER) -r $(CE_FOLDER) -d $(DICT_PATH) $(TEST_OPT)

wiki : dirs
	OMP_NUM_THREADS=4 $(PYTHON_INTERPRETER) $(MAIN_ENTRY) -t 5 -c $(CORPORA_FOLDER) -m $(MODELS_FOLDER) -b $(BENCHMARK_FOLDER) -r $(CE_FOLDER) $(TEST_OPT)

clean : 
	rm *.pyc *.py~

.PHONY : help dirs iter spec sort_comb wiki clean
