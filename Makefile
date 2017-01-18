PYTHON_INTERPRETER=python
MAIN_ENTRY=main.py
MODELS_FOLDER=/tmp/w2v/default_models/
USERNAME=`whoami`
CORPORA_FOLDER=/home/$(USERNAME)/workspace/master_thesis/data/
#CORPORA_FOLDER=/data1/alexandria/
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
	@echo '     make spec       train single model with specified configuration, default size=100 min_count=24'
	@echo '     make wiki       use wiki dump as corpora'
	@echo ''


dirs :
	mkdir -p MODELS_FOLDER
	mkdir -p CE_FOLDER
	
eval : dirs
	$(PYTHON_INTERPRETER) $(MAIN_ENTRY) -t 0 -f 0 -m $(MODELS_FOLDER) -b $(BENCHMARK_FOLDER) -r $(CE_FOLDER) $(TEST_OPT)

iter : dirs
	$(PYTHON_INTERPRETER) $(MAIN_ENTRY) -t 1 -f 0 -c $(CORPORA_FOLDER) -m $(MODELS_FOLDER) -b $(BENCHMARK_FOLDER) -r $(CE_FOLDER) -d $(DICT_PATH) $(TEST_OPT)

spec : dirs
	$(PYTHON_INTERPRETER) $(MAIN_ENTRY) -t 2 -f 0 -c $(CORPORA_FOLDER) -m $(MODELS_FOLDER) -b $(BENCHMARK_FOLDER) -r $(CE_FOLDER) -d $(DICT_PATH) $(TEST_OPT)

sort_comb : dirs
	$(PYTHON_INTERPRETER) $(MAIN_ENTRY) -t 3 -f 0 -c $(CORPORA_FOLDER) -m $(MODELS_FOLDER) -b $(BENCHMARK_FOLDER) -r $(CE_FOLDER) -d $(DICT_PATH) $(TEST_OPT)

wiki : dirs
	$(PYTHON_INTERPRETER) $(MAIN_ENTRY) -t 5 -c $(CORPORA_FOLDER) -m $(MODELS_FOLDER) -b $(BENCHMARK_FOLDER) -r $(CE_FOLDER) $(TEST_OPT)

.PHONY : help dirs iter spec sort_comb
