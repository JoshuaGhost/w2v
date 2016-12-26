class Evaluator(object):

	def eval_factory(self, benchmark_form, ce_dir, benchmark_dir,\
					 exp_type, ben_form, num_total_docs,\
					 num_sub_model, dim, min_count, test):
		if benchmark_form == 0:
			return Eval_wordnet_wiki(ce_dir, benchmark_dir,\
									 exp_type, ben_form, num_total_docs,\
									 num_sub_model, test)
