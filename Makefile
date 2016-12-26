PYTHON_INTERPRETER=python
MAIN_ENTRY=main.py
DIM_START=
MODELS_FOLDER=/tmp/w2v/default_models/
USERNAME=`whoami`
CORPORA_FOLDER=/home/$(USERNAME)/workspace/master_thesis/data/
BENCHMARK_FOLDER=/home/$(USERNAME)/workspace/master_thesis/benchmark/EN-WORDREP
CE_FOLDER=/tmp/w2v/default_results/
DICT_PATH=/usr/share/dict/words

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
	@echo ''


dirs :
	mkdir -p MODELS_FOLDER
	mkdir -p CE_FOLDER
	
iter : dirs
	$(PYTHON_INTERPRETER) $(MAIN_ENTRY) -t 1 -c $(CORPORA_FOLDER) -m $(MODELS_FOLDER) -b $(BENCHMARK_FOLDER) -r $(CE_FOLDER) -d $(DICT_PATH) -f 0 $(TEST_OPT)

.PHONY : help dirs iter