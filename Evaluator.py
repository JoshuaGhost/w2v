from Eval_wordnet_wiki import Eval_wordnet_wiki

class Evaluator(object):
    @classmethod
    def eval_factory(self, benchmark_form, ce_dir, benchmark_dir,\
                     exp_type, ben_form, num_total_docs,\
                     num_sub_model, test):
        if benchmark_form == 0:
            return Eval_wordnet_wiki(ce_dir, exp_type, benchmark_dir,\
                                     ben_form, num_total_docs, num_sub_model,\
                                     test)
        elif benchmark_form == 1:
            return Eval_benchmarks(ce_dir, exp_type, ben_form)
